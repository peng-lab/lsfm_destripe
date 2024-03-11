import torch


class cADAM(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        super(cADAM, self).__init__(
            params, defaults={"lr": lr, "b1": b1, "b2": b2, "eps": eps}
        )
        self.state = dict()
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = dict(
                    m=torch.zeros_like(p.data), v=torch.zeros_like(p.data)
                )
        self.i = 0

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state:
                    self.state[p] = dict(
                        m=torch.zeros_like(p.data), v=torch.zeros_like(p.data)
                    )
                m = self.state[p]["m"]
                v = self.state[p]["v"]
                m = (1 - group["b1"]) * p.grad.data + group["b1"] * m
                v = (1 - group["b2"]) * p.grad.data * torch.conj(p.grad.data) + group[
                    "b2"
                ] * v
                mhat = m / (1 - group["b1"] ** (self.i + 1))
                vhat = v / (1 - group["b2"] ** (self.i + 1))
                p.data -= group["lr"] * mhat / (torch.sqrt(vhat) + group["eps"])
                self.state[p]["m"] = m
                self.state[p]["v"] = v
        self.i += 1
