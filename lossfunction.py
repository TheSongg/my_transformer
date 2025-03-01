
class LossFunction:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        #  损失函数
        self.criterion = criterion
        #  优化器
        self.opt = opt

    def __call__(self, x, y, norm):
        #  将decoder的输出转换成概率
        x = self.generator(x)

        loss = self.criterion(x.contiguous().view(-1, x.size(1)), y.contiguous().view(-1)/norm)
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
