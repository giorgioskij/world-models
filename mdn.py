"""
A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
The code is taken and adapted from https://github.com/sagelywizard/pytorch-mdn
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math
from torch.nn import functional as F

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Linear(in_features, out_features * num_gaussians)
        # nn.Sequential(
        # nn.Linear(in_features, out_features * num_gaussians),
        # nn.Softmax(dim=-1),
        # )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        batch_size = minibatch.shape[1]
        pi = self.pi(minibatch)
        pi = pi.view(*pi.shape[:-1], self.num_gaussians, self.out_features)
        # softmax over the Gaussians
        pi = F.softmax(pi, dim=2)
        # sigma and mu need to be reshaped without flattening batch and sequence
        # -1  -->  *sigma.shape[:-1]
        sigma = torch.exp(self.sigma(minibatch))
        # sigma = self.sigma(minibatch)
        sigma = sigma.view(*sigma.shape[:-1], self.num_gaussians,
                           self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(*mu.shape[:-1], self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    # target needs to unsqueeze after batch, sequence dimensions
    target = target.unsqueeze(-2).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, -1)


def new_loss(logpi, logsigma, mu, target):
    # target = target.unsqueeze(-2).expand_as(logsigma)

    # logoneoversqrt2pi = math.log(ONEOVERSQRT2PI)
    # logsqrt2pi = math.log(math.sqrt(math.pi * 2))

    # logpi = logpi - torch.logsumexp(logpi, dim=-1, keepdim=True)  # normalize
    # n = -0.5 * ((target - mu) / torch.exp(logsigma))**2 - logsigma - logsqrt2pi
    # n = logpi + n
    # n = -torch.logsumexp(n, dim=-1)
    target = target.view(target.shape[0], target.shape[1], 1, target.shape[-1])

    epsilon = 1e-8
    result = torch.distributions.Normal(loc=mu, scale=logsigma)
    result = torch.exp(result.log_prob(target))
    result = torch.sum(result * logpi, dim=2)
    result = -torch.log(epsilon + result)
    result = result.mean(dim=-1)

    return result


def mdn_loss(pi, sigma, mu, target, lengths):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    lengths = lengths.to(target.device)

    # compute negative loglikelihood
    nll = new_loss(pi, sigma, mu, target)
    # prob = pi * gaussian_probability(sigma, mu, target)
    # nll = -torch.log(torch.sum(prob, dim=-1))

    # ignore padding
    mask = torch.ones_like(nll)
    mask = mask.t().cumsum(-1).t()
    mask = mask <= lengths
    mask = mask.long()
    nll = nll * mask
    # nll_old *= mask  # debug

    # mean is not correct because it would count also the padding
    # loss = torch.mean(nll)

    loss = torch.sum(nll) / torch.sum(mask)
    # loss_old = torch.sum(nll_old) / torch.sum(mask)

    # print(f'{loss=}')
    # print(f'{loss_old=}')

    return loss


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    pi = pi.permute(0, 2, 1)
    sigma = sigma.permute(0, 2, 1)
    mu = mu.permute(0, 2, 1)

    # Choose which gaussian we'll sample from
    chosen_gaussians = Categorical(pi).sample().view(*pi.shape[:-1], 1)
    gaussian_noise = torch.randn((sigma.shape[0], sigma.shape[1]),
                                 requires_grad=False).to(mu.device)

    # sample variances according to chosen gaussian
    variance_samples = sigma[:,
                             torch.arange(sigma.shape[1]),
                             chosen_gaussians.squeeze()].detach()
    # variance_samples = sigma.gather(1, pis).detach().squeeze()

    # sample means
    mean_samples = mu[:,
                      torch.arange(mu.shape[1]),
                      chosen_gaussians.squeeze()].detach()
    # mean_samples = mu.detach().gather(1, chosen_gaussians).squeeze()

    return (gaussian_noise * variance_samples + mean_samples)


def new_sample(pi, sigma, mu, temperature):
    pi = pi / temperature
    pi = pi - torch.logsumexp(pi, dim=-1, keepdim=True)  # normalize

    cat = torch.distributions.Categorical(logits=pi)

    # temperature
    sigs = torch.exp(sigma) * torch.sqrt(torch.tensor(temperature))
