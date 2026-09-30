import unittest
import torch
from run_round7 import shrink
from evaluate_round7 import choose

class ShrinkCheck(unittest.TestCase):
    def test_minimax_selection_uses_both_domains_and_fallback(self):
        values={'tau0':(3.,4.),'tau0.5':(2.9,3.95),'tau1':(2.5,4.1),
                'tau0_bias':(3.01,3.9),'tau0.5_bias':(3.02,3.8),'tau1_bias':(3.03,3.7)}
        run={k:{'wiki':{'ce':a},'ptb_seen':{'ce':b}} for k,(a,b) in values.items()}
        selected,_,_=choose([run,run,run])
        self.assertEqual(selected['weight_only'],'tau0.5')
        self.assertEqual(selected['hybrid'],'tau0')
        run['tau0_bias']={'wiki':{'ce':2.8},'ptb_seen':{'ce':4.1}}
        run['tau0.5_bias']={'wiki':{'ce':2.9},'ptb_seen':{'ce':3.9}}
        run['tau1_bias']={'wiki':{'ce':2.7},'ptb_seen':{'ce':4.2}}
        selected,_,_=choose([run,run,run])
        self.assertEqual(selected['hybrid'],'tau0.5_bias')
        self.assertEqual(selected['wiki_only_hybrid'],'tau1_bias')

    def test_grid_endpoints_and_bounded_code_movement(self):
        a=torch.arange(-8,8,dtype=torch.float32).reshape(4,4)
        b=torch.flip(a,[1]);scale=torch.tensor([.125,.25,.5,1.])[:,None]
        q0=a*scale;q1=b*scale
        torch.testing.assert_close(shrink(q0,q1,scale,0),q0,rtol=0,atol=0)
        torch.testing.assert_close(shrink(q0,q1,scale,1),q1,rtol=0,atol=0)
        middle=shrink(q0,q1,scale,.5)/scale
        torch.testing.assert_close(middle,middle.round(),rtol=0,atol=0)
        self.assertTrue(torch.all((middle-a).abs()<=(b-a).abs()))

if __name__=='__main__':unittest.main()
