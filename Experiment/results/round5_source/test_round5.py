import unittest
import torch
from run_round5 import fake_weight, prediction


class MergeObjectiveChecks(unittest.TestCase):
    def test_forward_grid_and_full_mlp_gradient(self):
        torch.manual_seed(27)
        code=torch.randn(5,3,requires_grad=True)
        scale=torch.rand(5,1)+.1
        x=torch.randn(3,7);gate=torch.randn(5,7);down=torch.randn(4,5)
        target=torch.randn(4,7)
        q=fake_weight(code,scale)
        torch.testing.assert_close(q,code.detach().round().clamp(-8,7)*scale,rtol=0,atol=0)
        estimate=prediction(q,x,gate,down,'mlp')
        loss=(estimate-target).square().mean()
        actual,=torch.autograd.grad(loss,code)
        expected=(2/target.numel())*((down.T@(estimate.detach()-target))*gate)@x.T*scale
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_gate_mismatch_changes_reconstruction_target(self):
        x=torch.ones(1,1);aq=torch.ones(1,1);down=torch.ones(1,1)
        up_teacher=torch.tensor([[3.]])
        product_teacher=torch.tensor([[6.]])
        branch_solution=up_teacher
        product_solution=product_teacher
        self.assertEqual(float((prediction(branch_solution,x,aq,down,'branch')-up_teacher).square()),0.)
        self.assertEqual(float((prediction(product_solution,x,aq,down,'mlp')-product_teacher).square()),0.)
        self.assertGreater(float((prediction(branch_solution,x,aq,down,'mlp')-product_teacher).square()),0.)

    def test_four_microbatches_match_effective_batch_gradient(self):
        torch.manual_seed(31)
        x=torch.randn(3,16);gate=torch.randn(5,16);down=torch.randn(4,5);y=torch.randn(4,16)
        code=torch.randn(5,3,requires_grad=True);scale=torch.rand(5,1)+.1
        full=(prediction(fake_weight(code,scale),x,gate,down,'mlp')-y).square().mean()
        expected,=torch.autograd.grad(full,code)
        for k in range(4):
            sl=slice(k*4,(k+1)*4)
            loss=(prediction(fake_weight(code,scale),x[:,sl],gate[:,sl],down,'mlp')-y[:,sl]).square().mean()/4
            loss.backward()
        torch.testing.assert_close(code.grad,expected,rtol=1e-5,atol=1e-6)


if __name__=='__main__':unittest.main()
