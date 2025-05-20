import numpy as np
from scipy.stats import rv_continuous

class ConditionalTailDist(rv_continuous):
    def __init__(self, x_hat, **kwargs):
        super().__init__(a=x_hat, name=f'conditional_tail_dist_{round(x_hat,3)}', **kwargs)
        self.x_hat = x_hat

    def _pdf(self, x, alpha, beta):
        """PDF for x >= x_hat."""
        x_hat = self.x_hat
        norm_const = np.exp(alpha * x_hat ** beta)
        return alpha * beta * x ** (beta - 1) * np.exp(-alpha * x ** beta) * norm_const

    def _logpdf(self, x, alpha, beta):
        x_hat = self.x_hat
        return (np.log(alpha) + np.log(beta) + (beta - 1) * np.log(x)
                - alpha * x ** beta + alpha * x_hat ** beta)

    def _sf(self, x, alpha, beta):
        """Survival function: P(X >= x | X >= x_hat)."""
        x_hat = self.x_hat
        return np.exp(-alpha * x ** beta + alpha * x_hat ** beta)

    def _logsf(self, x, alpha, beta):
        x_hat = self.x_hat
        return -alpha * x ** beta + alpha * x_hat ** beta
