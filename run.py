import datetime
import os
import sys
import tempfile

import multiprocess
import torch
import typer

from dpdl.cli import cli
from dpdl.device import distributed_backend, resolve_device, set_cuda_device
from dpdl.logger_config import configure_logger


def setup_torch():
    # Enable TensorFloat-32 for performance
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Reproducible results
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    # Fix Huggingface datasets map to work with multiple proceses.
    torch.set_num_threads(1)

    # Set to spawn so HF datasets map work with distributed
    multiprocess.set_start_method('spawn', force=True)


def _parse_device_arg(argv):
    for i, arg in enumerate(argv):
        if arg == '--device' and i + 1 < len(argv):
            return argv[i + 1]

        if arg.startswith('--device='):
            return arg.split('=', 1)[1]

    return 'auto'


def _resolve_distributed_env(log) -> tuple[int, int, int, str | None, str | None]:
    """
    Run in distributed if properly setup, e.g. by `torch.distributed.run` or `run_wraper.sh`.
    Otherwise default to single-process distributed mode.
    """
    world_size = os.getenv('WORLD_SIZE')
    rank = os.getenv('RANK')
    local_rank = os.getenv('LOCAL_RANK')

    dist_init_file = None
    init_method = None

    if world_size is None or local_rank is None or rank is None:
        log.info(
            "Distributed env vars 'WORLD_SIZE', 'RANK', and 'LOCAL_RANK' not set; "
            "defaulting to single-process distributed mode."
        )
        world_size = '1'
        rank = '0'
        local_rank = '0'

        os.environ['WORLD_SIZE'] = world_size
        os.environ['RANK'] = rank
        os.environ['LOCAL_RANK'] = local_rank

        # Let's communicate through a file socket
        dist_init_file = tempfile.NamedTemporaryFile(prefix='dpdl-dist-', suffix='.tmp', delete=False)
        dist_init_file.close()

        init_method = f'file://{dist_init_file.name}'

    return int(world_size), int(rank), int(local_rank), init_method, dist_init_file


def _init_process_group(device: torch.device, world_size: int, rank: int, init_method: str | None) -> None:
    init_kwargs = {
        'backend': distributed_backend(device),
        'world_size': world_size,
        'rank': rank,
    }

    if init_method is not None:
        init_kwargs['init_method'] = init_method

    if device.type == 'cuda':
        init_kwargs['device_id'] = torch.device('cuda', 0)  # Only one visible device

    torch.distributed.init_process_group(**init_kwargs)


def main():
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) == 1:
        try:
            typer.run(cli)
        except SystemExit as e:
            return int(e.code) if e.code is not None else 0

        return 0

    log = configure_logger()
    setup_torch()

    device_arg = os.getenv('DPDL_DEVICE') or _parse_device_arg(sys.argv)
    device = resolve_device(device_arg)

    world_size, rank, local_rank, init_method, dist_init_file = _resolve_distributed_env(log)

    log.info(
        f'Rank {rank} initializing - our world size is {world_size} and local rank is {local_rank}.'
    )

    # We only have one visible device exposed by `run_wrapper.sh` as recommended by AMD
    set_cuda_device(device)

    # Initialize the process group
    _init_process_group(device, world_size, rank, init_method)

    log.info(f'Rank {rank} initialized.')

    if torch.distributed.get_rank() == 0:
        log.info('All ranks initialized.')

    exit_code = 0
    fatal_error: BaseException | None = None

    # Run with CLI params and perform clean shutdown
    try:
        typer.run(cli)
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0
    except BaseException as exc:
        fatal_error = exc
        exit_code = 1
        log.exception(
            'Unhandled exception on rank %s. Tearing down distributed state without forcing syncrhonization.',
            rank,
        )
    finally:

        if torch.distributed.is_initialized():
            if fatal_error is None:
                try:
                    # Only do a synchronized barrier on the clean path.
                    # On fatal exceptions one or more ranks may already be unwinding,
                    # so a normal barrier tends to make teardown noisier rather than cleaner.
                    torch.distributed.barrier()
                except Exception:
                    pass

            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass

            log.info(f'Rank {rank} done!')

        if dist_init_file is not None:
            try:
                os.unlink(dist_init_file.name)
            except OSError:
                pass

    if fatal_error is not None:
        return exit_code

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
