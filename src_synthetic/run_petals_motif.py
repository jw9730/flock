import os
import pprint
import sys
from copy import deepcopy

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from motif import tasks, util
from motif.models import MOTIF

separator = ">" * 30
line = "-" * 30


def load_petals_dataset(cfg, device):
    """Load petals dataset."""
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning(f"Loading petals dataset from {cfg.dataset.root}")

    petals_dataset = []
    for i in range(220):
        dataset_root = os.path.expanduser(cfg.dataset.root)
        data_path = os.path.join(dataset_root, f"_{i}.pth")
        x = torch.load(data_path, weights_only=False)
        x.to(device)
        x.device = device
        petals_dataset.append(deepcopy(tasks.build_relation_hypergraph(x)))

    # Move data to device explicitly
    for data in petals_dataset:
        data.to(device)

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning(f"Loaded {len(petals_dataset)} graphs")
    return petals_dataset


def train_and_validate(cfg, model, data, device, logger):
    if cfg.train.num_epoch == 0:
        return
    world_size = util.get_world_size()
    rank = util.get_rank()

    # Prepare optimizer
    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    parallel_model.train()

    best_accuracy = float("-inf")
    best_epoch = -1
    for epoch in range(cfg.train.num_epoch):
        parallel_model.train()
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(f"Epoch {epoch} begin")

        epoch_losses = []

        for idx, test_data in enumerate(data):
            test_data.num_relations = torch.tensor(
                test_data.num_relations, device=device
            ).long()

            triplets = test_data.test_triplets  # [2, 3]

            # Create a batch with both triplets
            batch = triplets

            # For each triplet, compute predictions against all possible tails
            t_batch, _ = tasks.all_negative(test_data, batch)
            t_pred = parallel_model(test_data, t_batch)

            # Extract predictions for the specific tails we care about
            score_clean = t_pred[0, triplets[0, 1]]  # Score for clean half tail
            score_other = t_pred[1, triplets[1, 1]]  # Score for other half tail

            # Create targets: clean half should be 1, other half should be 0
            target_clean = torch.tensor(1.0, device=device)
            target_other = torch.tensor(0.0, device=device)

            # Compute binary cross entropy loss on the specific predictions
            loss_clean = F.binary_cross_entropy_with_logits(score_clean, target_clean)
            loss_other = F.binary_cross_entropy_with_logits(score_other, target_other)
            loss = loss_clean + loss_other

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

            if util.get_rank() == 0 and idx % cfg.train.log_interval == 0:
                logger.warning(separator)
                logger.warning(f"binary cross entropy: {loss}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning(f"Epoch {epoch} end")
            logger.warning(f"average binary cross entropy: {avg_loss:.4f}")
            logger.warning(f"Save checkpoint to model_epoch_{epoch}.pth")
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, f"model_epoch_{epoch}.pth")
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        _, accuracy = test(cfg, model, data, device, logger)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

    # Save final model
    if rank == 0:
        logger.warning(f"Load checkpoint from model_epoch_{best_epoch}.pth")
    state = torch.load(f"model_epoch_{best_epoch}.pth", map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, data, device, logger):
    """Test the model on all petals graphs to check fitting."""
    world_size = util.get_world_size()
    rank = util.get_rank()

    model.eval()

    success_count = 0
    total_count = len(data)

    correct_positive = 0  # Count of correct predictions for clean half triplets
    correct_negative = 0  # Count of correct predictions for other half triplets

    if rank == 0:
        logger.warning(separator)
        logger.warning("Testing model fit on training data")

    for test_data in tqdm(data):
        test_data.num_relations = torch.tensor(
            test_data.num_relations, device=device
        ).long()

        triplets = test_data.test_triplets  # [2, 3]
        batch = triplets

        # Get predictions for all possible tails
        t_batch, _ = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)

        # Get scores for the actual tails in the test triplets
        score_clean = t_pred[0, triplets[0, 1]].item()  # Score for clean half tail
        score_other = t_pred[1, triplets[1, 1]].item()  # Score for other half tail

        # Check if model distinguishes correctly
        if score_clean - score_other > 0.1:  # Using threshold of 0.1 for difference
            success_count += 1

        # Check individual predictions
        # For clean half: should predict high score
        if score_clean > 0:
            correct_positive += 1

        # For other half: should predict low score
        if score_other < 0:
            correct_negative += 1

    success_rate = success_count / total_count
    positive_accuracy = correct_positive / total_count
    negative_accuracy = correct_negative / total_count
    overall_accuracy = (correct_positive + correct_negative) / (2 * total_count)

    if rank == 0:
        logger.warning(separator)
        logger.warning(f"Success rate (clean > other): {success_rate:.2%}")
        logger.warning(f"True half accuracy: {positive_accuracy:.2%}")
        logger.warning(f"False half accuracy: {negative_accuracy:.2%}")
        logger.warning(f"Overall accuracy: {overall_accuracy:.2%}")

    return success_rate, overall_accuracy


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    task_name = cfg.task["name"]
    device = util.get_device(cfg)
    petals_dataset = load_petals_dataset(cfg, device)

    if cfg.model["class"] == "MOTIF":
        model = MOTIF(
            rel_model_cfg=cfg.model.relation_model,
            entity_model_cfg=cfg.model.entity_model,
        )
    else:
        raise NotImplementedError(
            f"Model class {cfg.model['class']} is not implemented"
        )

    model.to(device)

    if task_name != "SyntheticDatasetExperiment":
        raise ValueError(f"Unsupported task: {task_name}")

    # Train the model
    train_and_validate(cfg, model, petals_dataset, device, logger)

    # Test the model on the same data to check fitting
    success_rate, accuracy = test(cfg, model, petals_dataset, device, logger)

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Training and testing completed")
        logger.warning(f"Final success rate: {success_rate:.2%}")
        logger.warning(f"Final overall accuracy: {accuracy:.2%}")
