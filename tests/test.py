import torch

n, m = 128, 4

def main_worker(gpu):
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:12345', world_size=8, rank=gpu)
    torch.cuda.set_device(gpu)
    x = torch.randint(0, 2, (16,), device=f"cuda:{gpu}")

    # compute xTx using all_reduce
    #xTx = x.T @ x
    print("RANK: ", gpu, x)
    torch.distributed.all_reduce(x)
    if gpu == 0:
        print(gpu, x)

    exit()

    # compute xTx using all_gather
    ys = [torch.empty(n, m).cuda() for _ in range(2)]
    torch.distributed.all_gather(ys, x)
    y = torch.cat(ys)
    yTy = y.T @ y
    if gpu == 0:
        print(yTy)

if __name__ == '__main__':
    torch.multiprocessing.spawn(main_worker, nprocs=8)
