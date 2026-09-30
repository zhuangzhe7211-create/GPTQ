import unittest
import torch
from run_round6 import objective

class DirectionalObjectiveCheck(unittest.TestCase):
    def test_direction_gradient_and_mean_bias(self):
        torch.manual_seed(6)
        e=torch.randn(5,9,dtype=torch.double,requires_grad=True)
        g=torch.randn_like(e);u=g/g.norm(dim=0);s=torch.ones(9)
        loss=objective(e,'direction',s,u)
        actual,=torch.autograd.grad(loss,e)
        expected=2*e.detach()/e.numel()+2*u*(u*e.detach()).sum(0)/e.shape[1]
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(objective(e,'token',s,u),objective(e,'block',s,u))
        corrected=e.detach()-e.detach().mean(1,keepdim=True)
        torch.testing.assert_close(corrected.mean(1),torch.zeros(5,dtype=torch.double),atol=1e-15,rtol=0)
        self.assertLessEqual(float(corrected.square().mean()),float(e.detach().square().mean()))

if __name__=='__main__':unittest.main()
