import torch
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import TensorDataset

from bitnet_train.data import ResumableDataStream, make_loader


def _values(batch):
    value = batch[0] if isinstance(batch, (tuple, list)) else batch
    return tuple(int(item) for item in value.reshape(-1))


def test_resumable_stream_reproduces_next_batches_across_epoch_boundary():
    dataset = TensorDataset(torch.arange(23))
    continuous = ResumableDataStream(
        make_loader(dataset, batch_size=4, seed=71), seed=71)
    prefix = [_values(next(continuous)) for _ in range(7)]
    state = continuous.state_dict()
    expected = [_values(next(continuous)) for _ in range(8)]

    resumed = ResumableDataStream(
        make_loader(dataset, batch_size=4, seed=71), seed=71, state=state)
    assert [_values(next(resumed)) for _ in range(8)] == expected
    assert len(set(prefix + expected)) > 1
    assert resumed.state_dict() == continuous.state_dict()


def test_accelerate_two_rank_shards_resume_the_exact_local_stream():
    dataset = TensorDataset(torch.arange(48))
    for rank in (0, 1):
        def prepared():
            return prepare_data_loader(
                make_loader(dataset, batch_size=4, seed=19),
                num_processes=2, process_index=rank, split_batches=False,
                put_on_device=False)

        stream = ResumableDataStream(prepared(), seed=19)
        for _ in range(5):
            next(stream)
        state = stream.state_dict()
        expected = [_values(next(stream)) for _ in range(5)]
        resumed = ResumableDataStream(prepared(), seed=19, state=state)
        assert [_values(next(resumed)) for _ in range(5)] == expected


def test_stream_rejects_seed_or_epoch_length_mismatch():
    dataset = TensorDataset(torch.arange(24))
    stream = ResumableDataStream(make_loader(dataset, 4, 3), seed=3)
    state = stream.state_dict()
    try:
        ResumableDataStream(make_loader(dataset, 4, 4), seed=4, state=state)
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("seed mismatch was accepted")
