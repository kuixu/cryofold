# Copyright (c) CryoNet Team, and its affiliates. All Rights Reserved

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os, json, copy
import subprocess
import time
import math
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List, TypeVar, Iterator, Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import torch.nn.functional as F

T_co = TypeVar('T_co', covariant=True)

# from .tvops import output_size #, #_new_empty_tensor
# from util import timeutil

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def update_status(jobid, jdict):
    sfile = 'data/jobs/{}/status'.format(jobid)
    jdata = json.load(open(sfile))
    jdata.update(jdict)
    json.dump(jdata, open(sfile, 'w'), indent=4, cls=NpEncoder)

    # with open(sfile, "w") as f:
    #         f.write(json.dumps(jdata))

def to_gpu(data, dtype=torch.float32):
    for k in data:
        if not isinstance(data[k], str):
            data[k] = data[k].to(dtype).cuda()



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), "key: {} error".format(k)
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        aux_str = []
        meters = { k:self.meters[k] for k in sorted(self.meters.keys())} 
        for name, meter in meters.items():
            if "unscaled" in name:
                continue
            if len(name.split('_'))<=2:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
            # else:
            #     aux_str.append(
            #         "{}: {}".format(name, str(meter))
            #     )
        loss_str.append("\n")
        loss_str.extend(aux_str)

        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, iterable, print_freq, header=None, rank=0, stepping=False):
        i = 0
        if not header:
            header = ''
        if stepping:
            start_time = time.time()
            end = time.time()
            iter_time = SmoothedValue(fmt='{avg:.4f}')
            data_time = SmoothedValue(fmt='{avg:.4f}')
            space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'GMem: {memory:.2f}GB',
                '\n\n'
            ])
        
        MB = 1024.0 * 1024.0
        GB = MB * 1024.0
        for obj in iterable:
            # print(timeutil.now(), "obj:", len(obj))
            if stepping:
                data_time.update(time.time() - end)
            yield obj
            if rank!=0:
                continue
            if stepping:
                iter_time.update(time.time() - end)
            if stepping and (i % print_freq == 0 or i == len(iterable) - 1):
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(log_msg.format(
                    i+1, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / GB))
                
            i += 1
            if stepping:
                end = time.time()
        if stepping:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            if rank==0:
                print('{} Total time: {} ({:.4f} s / it)'.format(
                    header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    # print(len(batch))
    if isinstance(batch[1][0], int):
        del batch[0]
        ind = 1
    if isinstance(batch[1][0], int):
        del batch[0]
        ind = 1
    if isinstance(batch[1][0], int):
        del batch[0]
        ind = 1
    if isinstance(batch[1][0], int):
        del batch[0]
        ind = 1
    if len(batch) ==2:
        ind = 0
    # import pdb;pdb.set_trace()
    # batch[0] = nested_tensor_from_tensor_list(batch[0])
    # if isinstance(batch[ind][0], int):
    #     del batch[0]
    if len(batch[ind][0].shape) == 4:
        batch[ind] = nested_tensor_from_tensor_list(batch[ind])
    else:
        print("wrong batch shape: ", len(batch))
    return tuple(batch)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    # if isinstance(tensor_list[0], int):
    #     import pdb;pdb.set_trace()
    
    # TODO make it support different-sized images
    # max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
    # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    max_size = tuple(tensor_list[0].shape )
    batch_shape = (len(tensor_list),) + max_size
    # print("batch_shape:", batch_shape)
    b, c, w, h, l = batch_shape
    dtype  = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    # mask   = torch.zeros_like(batch_shape, dtype=torch.bool, device=device)
    mask   = None
    # tensor[0] = tensor_list[0]
    # print(timeutil.now(), "copy_...")
    for img, pad_img in zip(tensor_list, tensor):
        pad_img.copy_(img)
        # m[: img.shape[1], :img.shape[2]] = False
    # print(timeutil.now(), "copy end.")
    return NestedTensor(tensor, mask)

class NestedTensor(object):
    def __init__(self, tensors, mask=None):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def set_mask(self, mask):
        self.mask = mask

    def decompose(self):
        return self.tensors, self.mask

    def sub(self, i=0, s=1):
        assert i < self.tensors.shape[1], "NestedTensor size error."
        return NestedTensor(self.tensors[:,i:i+s], self.mask)

    def reshape_for_infer(self):
        s = self.tensors.shape
        return NestedTensor(self.tensors.view(s[1],s[0],s[2],s[3],s[4],), self.mask)
    
    def to_infer(self):
        s = self.tensors.shape
        return NestedTensor(self.tensors.view(s[1],s[0],s[2],s[3],s[4],), self.mask)

    @property
    def shape(self):
        return self.tensors.shape

    def __repr__(self):
        return repr(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    if  args.dist_backend=='pytorch':
        args.world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank() 
        device = rank % args.nproc
        args.gpu = device
        args.rank = rank
        torch.cuda.set_device(device)
        if args.rank:
            print("dist_backend: pytorch nccl")
    elif  args.dist_backend=='pytorch_gloo':
        args.world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', init_method='env://')
        rank = dist.get_rank() 
        device = rank % args.nproc
        args.gpu = device
        args.rank = rank
        torch.cuda.set_device(device)
        if args.rank:
            print("dist_backend: pytorch gloo")
    elif 'SLURM_PROCID' in os.environ and args.dist_backend=='slurm':
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        _init_dist_slurm(backend='nccl', port=29505)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    

def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)




def _init_dist_slurm(backend, port=29503, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    
    torch.cuda.set_device(proc_id % num_gpus)

    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    try:
        dist.init_process_group(backend=backend)
    except RuntimeError as e:
        print("port: ", port, e)
        _init_dist_slurm(backend=backend, port=port+1)
    


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # if get_rank() == 0:
    #     # import pdb; pdb.set_trace()
    #     v, c = torch.unique(pred, return_counts=True)
    #     v1, c1 = torch.unique(target, return_counts=True)
    #     print("v, c:", v, c, v1, c1)
        # print("v, c:", v, c)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
                    input, size, scale_factor, mode, align_corners
                )
    output_shape = output_size(2, input, size, scale_factor)
    output_shape = list(input.shape[:-3]) + list(output_shape)
    return _new_empty_tensor(input, output_shape)

def load_model(args, model_without_ddp, optimizer, lr_scheduler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            pp = Path(args.resume)
            if pp.exists():
                if args.rank == 0:
                    print("loading checkpoint file: {}.".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                # loading model version
                if args.rank == 0 and "version" in checkpoint:
                    args.model_version = args.model_version + checkpoint["version"]
                    print("Version: ", checkpoint["version"])
                # loading model
                missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
                if len(missing_keys) > 0:
                    if args.rank == 0:
                        print('Missing Keys: {}'.format(missing_keys))
                if len(unexpected_keys) > 0:
                    if args.rank == 0:
                        print('Unexpected Keys: {}'.format(unexpected_keys))
                # loading optimizer
                if not args.infer and not args.no_optimizer :
                    if 'optimizer' in checkpoint:
                        p_groups = copy.deepcopy(optimizer.param_groups)
                        try:
                            optimizer.load_state_dict(checkpoint['optimizer'])
                            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                                pg['lr'] = pg_old['lr']
                                pg['initial_lr'] = pg_old['initial_lr']
                            # if args.rank == 0:
                            #     print(optimizer.param_groups)
                            
                            if args.rank == 0:
                                print("loading optimizer in checkpoint file: {}.".format(args.resume))
                        except ValueError:
                            if args.rank == 0:
                                print("failed loading optimizer in checkpoint file: {}.".format(args.resume))
                # loading lr_scheduler
                if not args.infer  and not args.no_lr_scheduler and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                    args.override_resumed_lr_drop = True
                    if args.override_resumed_lr_drop:
                        if args.rank == 0:
                            print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                        lr_scheduler.step_size = args.lr_drop
                        lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                    if args.override_resumed_lr_base:
                        if args.rank == 0:
                            print('Warning: (hack) args.override_resumed_lr_base is set to True, so args.lr would override lr_drop in resumed lr_scheduler.')
                        lr_scheduler.base_lrs = list(map(lambda group: args.lr, optimizer.param_groups))
                    lr_scheduler.step(lr_scheduler.last_epoch)
                    args.start_epoch = checkpoint['epoch'] + 1
                else:
                    args.start_epoch = 1
                    
            else:
                if args.rank == 0:
                    print("no checkpoint file: {}.".format(args.resume))

class AdamW(torch.optim.Optimizer):
    r"""Implements AdamW algorithm.
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    # state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    .. warning::
        In distributed mode, calling the ``set_epoch`` method is needed to
        make shuffling work; each process will use the same random seed
        otherwise.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch




class DistributedWeightedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Args:
        dataset: Dataset used for sampling.
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, weights: Sequence[float],
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if weights is not None:
            assert len(dataset) == len(weights), f"Dataset size {len(dataset)} != weights size {len(weights)}"
            self.weights = torch.as_tensor(weights, dtype=torch.double)
        else:
            self.weights = None
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            if self.weights is None:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = torch.multinomial(self.weights, len(self.dataset), replacement=True, generator=g).cpu().tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



class SamplerWeightScheduler:
    """
    Dataloader sampler scheduler
    
    import torch
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    from dist_weighted_sampler import DistributedWeightedSampler
    from sampler_weight_scheduler import SamplerWeightScheduler
    from torch.distributed import get_world_size, get_rank, init_process_group, all_reduce, ReduceOp
    
    ## Warp the dataset to provide the sample index
    class Dataset_Wrapper(datasets.FashionMNIST):
        def __getitem__(self, index: int):
            sample = super().__getitem__(index)
            if isinstance(sample, list) or isinstance(sample, tuple):
                sample = list(sample)
            else:
                sample = [ sample ]
            return [index] + sample
    train_dataset = Dataset_Wrapper(root="data", train=False, download=True, transform=ToTensor())
    sampler = DistributedWeightedSampler(train_dataset, weights=None, shuffle=True, drop_last=True)
    sampler_scheduler = SamplerWeightScheduler(sampler)
    dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, sampler = sampler)
    for epoch in range(10):
        for batch_idx, batch_inp, batch_label in dataloader:
            batch_size = len(batch_idx)
            random_loss = torch.rand(batch_size)
            sampler_scheduler.step(batch_idx, random_loss)
        sampler_scheduler.update()
    """
    def __init__(self, sampler, init_sample_rate=10.0, min_sample_rate=1.0, max_sample_rate=100.0, verbose=False):
        self.sampler = sampler
        self.init_sample_rate = init_sample_rate
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.verbose = verbose
        
        self.sampler.weight   = torch.ones( len(self.sampler.dataset ) ) * init_sample_rate
        self.epoch_loss_value = torch.zeros( len(self.sampler.dataset ) ).float().cuda()
        self.epoch_loss_num   = torch.zeros( len(self.sampler.dataset ) ).float().cuda()
    
    def step(self, idx, loss):
        """
        idx: [N_batch]
        loss: [N_batch]
        """
        # assert isinstance(idx, torch.Tensor)
        assert isinstance(loss, torch.Tensor)
        # assert idx.ndim == 1
        # assert loss.ndim == 1

        self.epoch_loss_value[idx] += loss.detach().cuda()
        self.epoch_loss_num[idx] += 1.0
    
    def sync(self):
        if torch.distributed.is_initialized() and get_world_size() > 1:
            dist.all_reduce(self.epoch_loss_value, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.epoch_loss_num, op=dist.ReduceOp.SUM)
    
    def update(self):
        self.sync()
        self.epoch_loss_value /= self.epoch_loss_num + 1e-5
        mean_loss = self.epoch_loss_value[ self.epoch_loss_num > 0 ].median()
        multiplier = self.epoch_loss_value / (mean_loss + 1e-5)
        multiplier[ self.epoch_loss_num == 0 ] = 1.0
        self.sampler.weight *= multiplier.to(self.sampler.weight)
        self.sampler.weight = torch.clip( self.sampler.weight, self.min_sample_rate, self.max_sample_rate )
        if self.verbose:
            print( f"Update weights with multiplier: {multiplier}" )
            print( f"New weights: {self.sampler.weight}" )
        
        self.epoch_loss_value[:] = 0.0
        self.epoch_loss_num[:] = 0.0
    
    def load_state_dict(self, state_dict):
        assert isinstance(state_dict['weight'], torch.Tensor)
        self.sampler.weight = state_dict['weight']
        self.sampler.weight = torch.clip( self.sampler.weight, self.min_sample_rate, self.max_sample_rate )
    
    def state_dict(self):
        return {
            'weight': self.sampler.weight
        }
