import unittest, omnical.info as Oi, omnical._omnical as _O
import omnical.calibration_omni as omni
import numpy as np
import os

redinfo_psa32 = os.path.dirname(os.path.realpath(__file__)) + '/../doc/redundantinfo_PSA32.txt'
tmpfile = 'aiwvlkasfdlk'
tmpnpz = 'xcmowtpqpow.npz'
VERBOSE = False

class TestRedInfo(unittest.TestCase):
    def tearDown(self):
        if os.path.exists(tmpfile): os.remove(tmpfile)
        if os.path.exists(tmpnpz): os.remove(tmpnpz)
    def test_init(self):
        i = Oi.RedundantInfoLegacy(threshold=64)
        self.assertEqual(i.threshold, 64)
        class RedInf(Oi.RedundantInfo):
            testcase = False
            def from_npz(self, filename):
                self.testcase = True
        i = RedInf(filename='whatever')
        self.assertTrue(i.testcase)
    def test_basic_getsetitem(self):
        i = Oi.RedundantInfo()
        i.nAntenna = 32
        self.assertEqual(i.nAntenna, 32)
        self.assertEqual(i['nAntenna'], 32)
        i['nAntenna'] = 64
        self.assertEqual(i.nAntenna, 64)
        self.assertEqual(i['nAntenna'], 64)
    def test_compare(self):
        i = Oi.RedundantInfoLegacy()
        self.assertTrue(i.compare(i, verbose=VERBOSE))
    def test_init_from_reds(self):
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        reds = [[(0,1),(1,2),(2,3)],[(0,2),(1,3)]]
        i = Oi.RedundantInfo()
        i.init_from_reds(reds,antpos)
        self.assertEqual(i.nAntenna,4)
        self.assertTrue(np.all(i.antloc == antpos))
        self.assertEqual(i.nBaseline,5)
        self.assertEqual(i.ublcount[0],3)
        self.assertEqual(i.ublcount[1],2)
        self.assertTrue(np.all(i.ublindex == np.arange(5,dtype=np.int32)))
        self.assertTrue(np.all(i.ubl[0] == np.array([1.,0,0],dtype=np.float32)))
        self.assertTrue(np.all(i.ubl[1] == np.array([2.,0,0],dtype=np.float32)))
        self.assertTrue(i.blperant[0] == 2)
        self.assertTrue(i.blperant[1] == 3)
        self.assertTrue(i.blperant[2] == 3)
        self.assertTrue(i.blperant[3] == 2)
    def test_bl_order(self):
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        reds = [[(0,1),(1,2),(2,3)],[(0,2),(1,3)]]
        i = Oi.RedundantInfoLegacy()
        i.init_from_reds(reds,antpos)
        self.assertEqual(i.bl_order(), [bl for ublgp in reds for bl in ublgp])
        i = Oi.RedundantInfoLegacy()
        i.fromfile_txt(redinfo_psa32)
        self.assertEqual(i.bl_order()[:5], [(0,4),(1,4),(2,4),(3,4),(0,5)])
    def test_order_data(self):
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        reds = [[(0,1),(1,2),(2,3)],[(0,2),(1,3)]]
        i = Oi.RedundantInfo()
        i.init_from_reds(reds,antpos)
        dd = {
            (0,1):np.array([[0,1j]]),
            (1,2):np.array([[0,1j]]),
            (2,3):np.array([[0,1j]]),
            (2,0):np.array([[0,1j]]),
            (1,3):np.array([[0,1j]]),
        }
        d = i.order_data(dd)
        self.assertTrue(np.all(d[...,0] == np.array([[0,1j]])))
        self.assertTrue(np.all(d[...,1] == np.array([[0,1j]])))
        self.assertTrue(np.all(d[...,2] == np.array([[0,1j]])))
        self.assertTrue(np.all(d[...,3] == np.array([[0,1j]]).conj()))
        self.assertTrue(np.all(d[...,4] == np.array([[0,1j]])))
    def test_get_reds(self):
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        reds = [[(0,1),(1,2),(2,3)],[(0,2),(1,3)]]
        i = Oi.RedundantInfo()
        i.init_from_reds(reds,antpos)
        reds2 = i.get_reds()
        self.assertEqual(reds, reds2)
    def test_tofrom_reds(self):
        i1 = Oi.RedundantInfoLegacy()
        i1.fromfile_txt(redinfo_psa32)
        reds = i1.get_reds()
        antpos = i1.get_antpos()
        i2 = Oi.RedundantInfo()
        i2.init_from_reds(reds, antpos)
        self.assertTrue(np.all(i1.antloc == i2.antloc))
        self.assertTrue(np.all(i1.bl2d[i1.ublindex] == i2.bl2d[i2.ublindex]))
        #self.assertTrue(np.allclose(i1.ubl, i2.ubl, 1e-4))
        #import IPython; IPython.embed()
        #self.assertTrue(i1.compare(i2, tol=1e-3, verbose=VERBOSE)) # XXX this won't work b/c baselines reordered
    #def test_fromfiletxt(self):
    #    i1 = omni.read_redundantinfo_txt(redinfo_psa32)
    #    i1 = omni.RedundantInfo(i1)
    #    i2 = Oi.RedundantInfo()
    #    i2.fromfile_txt(redinfo_psa32)
    #    self.assertTrue(i2.compare(i1,verbose=VERBOSE))
    #def test_tofromarray(self):
    #    i1 = Oi.RedundantInfoLegacy()
    #    i1.fromfile_txt(redinfo_psa32)
    #    i2 = Oi.RedundantInfoLegacy()
    #    d = i1.to_array(verbose=VERBOSE)
    #    # XXX yuck
    #    d = np.array(d)
    #    markerindex = np.where(d == Oi.MARKER)[0]
    #    d = np.array([np.array(d[markerindex[i]+1:markerindex[i+1]]) for i in range(len(markerindex)-1)])
    #    i2.from_array(d)
    #    i2.update()
    #    self.assertTrue(i1.compare(i2, verbose=VERBOSE))
    #def test_tofromfile(self):
    #    i1 = Oi.RedundantInfoLegacy()
    #    i1.fromfile_txt(redinfo_psa32)
    #    i2 = Oi.RedundantInfoLegacy()
    #    i1.tofile(tmpfile)
    #    i2.fromfile(tmpfile)
    #    self.assertTrue(i1.compare(i2, verbose=VERBOSE))
    def test_tofromnpz(self):
        i1 = Oi.RedundantInfoLegacy()
        i1.fromfile_txt(redinfo_psa32)
        i2 = Oi.RedundantInfo()
        i1.to_npz(tmpnpz)
        i2.from_npz(tmpnpz)
        self.assertTrue(i1.get_reds() == i2.get_reds())


class TestFCRedInfo(TestRedInfo):
    def test_init_from_reds(self):
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        reds = [[(0,1),(1,2),(2,3)],[(0,2),(1,3)]]
        blpairs = [((0,1),(1,2)),((0,1),(2,3)),((1,2),(2,3)),((0,2),(1,3))]
        A = np.array([[1, -2, 1, 0], [1,-1,-1,1], [0,1,-2,1], [1,-1,-1,1]])
        i = Oi.FirstCalRedundantInfo()
        i.init_from_reds(reds,antpos)
        self.assertTrue(np.all(i.subsetant == np.arange(4, dtype=np.int32)))
        self.assertEqual(i.reds,reds)
        self.assertEqual(i.bl_pairs,blpairs)
        self.assertTrue(i.blperant[0] == 2)
        self.assertTrue(i.blperant[1] == 3)
        self.assertTrue(i.blperant[2] == 3)
        self.assertTrue(i.blperant[3] == 2)
        self.assertTrue(np.all(i.A == A))
    def test_bl_index(self):
        reds = [[(0,1),(1,2),(2,3)],[(0,2),(1,3)]]
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        i = Oi.FirstCalRedundantInfo()
        i.init_from_reds(reds,antpos)
        bls_order = [bl for ublgp in reds for bl in ublgp]
        for k,b in enumerate(bls_order):
            self.assertEqual(i.bl_index(b),k)
    def test_blpair_index(self):
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        reds = [[(0,1),(1,2),(2,3)],[(0,2),(1,3)]]
        blpairs = [((0,1),(1,2)),((0,1),(2,3)),((1,2),(2,3)),((0,2),(1,3))]
        i = Oi.FirstCalRedundantInfo()
        i.init_from_reds(reds,antpos)
        for k,bp in enumerate(blpairs):
            self.assertEqual(i.blpair_index(bp), k)
    def test_blpair2antindex(self):
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        reds = [[(0,1),(1,2),(2,3)],[(0,2),(1,3)]]
        blpairs = [((0,1),(1,2)),((0,1),(2,3)),((1,2),(2,3)),((0,2),(1,3))]
        i = Oi.FirstCalRedundantInfo()
        i.init_from_reds(reds,antpos)
        for bp in blpairs:
            self.assertTrue(np.all(i.blpair2antind(bp) == map(i.ant_index,np.array(bp).flatten())))
         
        
             
        


    

if __name__ == '__main__':
    unittest.main()
