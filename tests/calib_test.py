import omnical.info as Oi, omnical.calib as Oc, omnical._omnical as _O
#import omnical.calibration_omni as omni
import numpy as np, numpy.linalg as la
import os, unittest
import nose.tools as nt

redinfo_psa32 = os.path.dirname(os.path.realpath(__file__)) + '/../doc/redundantinfo_PSA32.txt'
#infotestpath = os.path.dirname(os.path.realpath(__file__)) + '/redundantinfo_test.bin'
infotestpath = os.path.dirname(os.path.realpath(__file__)) + '/calib_test_redinfo.npz'
testdata = os.path.dirname(os.path.realpath(__file__)) + '/testinfo/calib_test_data_%02d.npz'

VERBOSE = False

class TestMethods(unittest.TestCase):
    def setUp(self):
        self.info = Oi.RedundantInfoLegacy(filename=redinfo_psa32, txtmode=True)

        self.info2 = Oi.RedundantInfo()
        self.info2.init_from_reds([[(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9)],
                             [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9)],
                             [(0, 6), (1, 7), (2, 8), (3, 9)],
                             [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)],
                             [(0, 8), (1, 9)],
                             [(0, 7), (1, 8), (2, 9)],
                             [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9)],
                             [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]],
                             np.array([[0., 0., 1.],
                                       [ 0., 50., 1.],
                                       [ 0., 100., 1.],
                                       [ 0., 150., 1.],
                                       [ 0., 200., 1.],
                                       [ 0., 250., 1.],
                                       [ 0., 300., 1.],
                                       [ 0., 350., 1.],
                                       [ 0., 400., 1.],
                                       [ 0., 450., 1.]]))

        self.freqs = np.linspace(.1, .2, 16)
        self.times = np.arange(4)
        self.reds = self.info2.get_reds()
        self.true_vis = {}
        for i, rg in enumerate(self.reds):
            rd = np.array(np.random.randn(self.times.size, self.freqs.size) + 1j * np.random.randn(self.times.size, self.freqs.size), dtype=np.complex64)
            self.true_vis[rg[0]] = rd
        self.true_gains = {i: np.ones((self.times.size, self.freqs.size), dtype=np.complex64) for i in self.info2.subsetant}  # make it more complicated
        self.data = {}
        self.bl2red = {}
        for rg in self.reds:
            for r in rg:
                self.bl2red[r] = rg[0]
        for redgp in self.reds:
            for ai, aj in redgp:
                self.data[ai, aj] = self.true_vis[self.bl2red[ai, aj]] * self.true_gains[ai] * np.conj(self.true_gains[aj])
        self.unitgains = {ant: np.ones((self.times.size, self.freqs.size), dtype=np.complex64) for ant in self.info2.subsetant}

    def test_pack_calpar(self):
        calpar = np.zeros((2,3,Oc.calpar_size(self.info.nAntenna, len(self.info.ublcount))), dtype=np.float32)
        self.assertTrue(np.all(Oc.pack_calpar(self.info,calpar) == 0))
        self.assertRaises(AssertionError, Oc.pack_calpar, self.info, calpar[...,:-1])
        bp = np.array([[1+2j,3+4j,5+6j],[2+1j,4+3j,6+5j]])
        amp,phs = np.log10(np.abs(bp)), np.angle(bp)
        gains = {0:bp}
        Oc.pack_calpar(self.info,calpar,gains=gains)
        self.assertTrue(np.allclose(calpar[...,3+0], amp))
        self.assertTrue(np.allclose(calpar[...,32+3+0],phs))
        calpar *= 0
        gains = {1:bp[0]}
        Oc.pack_calpar(self.info,calpar,gains=gains)
        self.assertTrue(np.allclose(calpar[0,:,3+1], amp[0]))
        self.assertTrue(np.allclose(calpar[1,:,3+1], amp[0]))
        self.assertTrue(np.allclose(calpar[0,:,32+3+1],phs[0]))
        self.assertTrue(np.allclose(calpar[1,:,32+3+1],phs[0]))
        vis = {(0,16):bp}
        Oc.pack_calpar(self.info,calpar,vis=vis)
        self.assertTrue(np.allclose(calpar[...,3+2*32+2*12], bp.real))
        self.assertTrue(np.allclose(calpar[...,3+2*32+2*12+1], bp.imag))
    def test_unpack_calpar(self):
        calpar = np.zeros((2,3,Oc.calpar_size(self.info.nAntenna, len(self.info.ublcount))), dtype=np.float32)
        m,g,v = Oc.unpack_calpar(self.info,calpar)
        antchisq = [k for k in m if k.startswith('chisq') and len(k) > len('chisq')]
        self.assertEqual(m['iter'].shape, (2,3))
        self.assertEqual(len(antchisq), self.info.nAntenna)
        self.assertTrue(np.all(m['iter'] == 0))
        self.assertTrue(np.all(m['chisq'] == 0))
        for k in antchisq:
            self.assertTrue(np.all(m[k] == 0))
        self.assertEqual(len(g), 32)
        for i in xrange(32):
            self.assertTrue(np.all(g[i] == 1)) # 1 b/c 10**0 = 1
        self.assertEqual(len(v), len(self.info.ublcount))
        ubls = {}
        for i,j in v:
            n = self.info.bl1dmatrix[i,j]
            ubls[self.info.bltoubl[n]] = n
        for u in xrange(len(self.info.ublcount)):
            self.assertTrue(ubls.has_key(u))
    def test_redcal(self):
        #check that logcal give 0 chi2 for all 20 testinfos
        for index in xrange(20):
            arrayinfopath = os.path.dirname(os.path.realpath(__file__)) + '/testinfo/test'+str(index+1)+'_array_info.txt'
            c = Oc.RedundantCalibrator(56)
            c.compute_redundantinfo(arrayinfopath, tol=.1)
            info = c.Info
            npz = np.load(testdata % index)
            bls = [tuple(bl) for bl in npz['bls']]
            dd = dict(zip(bls, npz['vis']))
            m,g,v = Oc.redcal(dd, info, removedegen=True,maxiter=50,stepsize=.2,computeUBLFit=True,conv=1e-5,uselogcal=True)
            calparpath = os.path.dirname(os.path.realpath(__file__)) + '/testinfo/test'+str(index+1)+'_calpar.txt'
            with open(calparpath) as f:
                rawinfo = [[float(x) for x in line.split()] for line in f]
            temp = np.array(rawinfo[:-1])
            correctcalpar = (np.array(temp[:,0]) + 1.0j*np.array(temp[:,1]))
            i = g.keys()[0]
            scalar = correctcalpar[i].real / g[i].real
            for i in xrange(56):
                if not g.has_key(i): continue
                self.assertAlmostEqual(np.abs(correctcalpar[i] - g[i] * scalar), 0, 4)

    def test_unitgains(self):
        nt.assert_equal(np.testing.assert_equal(Oc.create_unitgains(self.data),
                       {ant: np.ones((self.times.size, self.freqs.size),
                                      dtype=np.complex64) for ant in self.info2.subsetant}), None)

    def test_logcal(self):
        m, g, v = Oc.logcal(self.data, self.info2, gainstart=self.unitgains)
        nt.assert_equal(np.testing.assert_equal(g, self.unitgains), None)

    def test_lincal(self):
        m1, g1, v1 = Oc.logcal(self.data, self.info2, gainstart=self.unitgains)
        m, g, v = Oc.lincal(self.data, self.info2, gainstart=g1, visstart=v1)
        nt.assert_equal(np.testing.assert_equal(g, self.unitgains), None)

    def test_redcal_degeneracies(self):
        m1, g1, v1 = Oc.logcal(self.data, self.info2)
        m, g, v = Oc.lincal(self.data, self.info2, gainstart=g1, visstart=v1)
        _, g, v = Oc.removedegen(self.info2, g, v, Oc.create_unitgains(self.data))
        nt.assert_equal(np.testing.assert_almost_equal(m['chisq'], np.zeros_like(m['chisq']), decimal=8), None)

        # make sure model visibilities equals true visibilities
        for bl in v.keys():
            nt.assert_equal(np.testing.assert_almost_equal(v[bl], self.true_vis[bl], decimal=8), None)

        # make sure gains equal true gains
        for ai in g.keys():
            nt.assert_equal(np.testing.assert_almost_equal(g[ai], self.true_gains[ai], decimal=8), None)

        # test to make sure degeneracies keep average amplitudes and phases constant.
        gains = np.array([g[i] for i in g.keys()])
        nt.assert_equal(np.testing.assert_almost_equal(np.mean(np.abs(gains), axis=0), np.ones_like(gains), decimal=8))
        nt.assert_equal(np.testing.assert_almost_equal(np.mean(np.angle(gains), axis=0), np.zeros_like(np.real(gains[0])), decimal=8), None)

    def test_redcal_xtalk(self):
        antpos = np.array([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        d = {(1,2): np.array([[1.]], dtype=np.complex64), (2,3): np.array([[1.+1j]], dtype=np.complex64)}
        x = {(1,2): np.array([[0.]], dtype=np.complex64), (2,3): np.array([[0.+1j]], dtype=np.complex64)}
        reds = [[(1,2),(2,3)]]
        info = Oi.RedundantInfo(); info.init_from_reds(reds, antpos)
        m,g,v = Oc.redcal(d, info, xtalk=x, uselogcal=False)
        self.assertEqual(g[1][0,0], 1.)
        self.assertEqual(g[2][0,0], 1.)
        self.assertEqual(g[3][0,0], 1.)
        #2D array testing
        d = {(1,2): np.array([[1.,2],[3.,4]],dtype=np.complex64), (2,3): np.array([[1.+1j,2+2j],[3+3j,4+4j]],dtype=np.complex64)}
        x = {(1,2): np.array([[0.,2],[2.,3]],dtype=np.complex64), (2,3): np.array([[0.+1j,1+2j],[2+3j,3+4j]],dtype=np.complex64)}
        m,g,v = Oc.redcal(d, info, xtalk=x, uselogcal=False)
        self.assertEqual(g[1][0,0], 1.)
        self.assertEqual(g[2][0,0], 1.)
        self.assertEqual(g[3][0,0], 1.)
        self.assertEqual(m['res'][(2,3)][0][0],0.)


class TestRedCal(unittest.TestCase):
    #def setUp(self):
    #    self.i = Oi.RedundantInfo()
    #    self.i.fromfile_txt(redinfo_psa32)
    def tearDown(self):
        if os.path.exists(infotestpath): os.remove(infotestpath)

    def test_large_info_IO(self):
        calibrator = Oc.RedundantCalibrator(150)
        calibrator.compute_redundantinfo()
        calibrator.write_redundantinfo(infotestpath, verbose=VERBOSE)
        info2 = Oi.RedundantInfo(filename=infotestpath)
        self.assertEqual(calibrator.Info.nAntenna, info2.nAntenna)
        self.assertEqual(calibrator.Info.nBaseline, info2.nBaseline)
        self.assertEqual(calibrator.Info.get_reds(), info2.get_reds())
        os.remove(infotestpath)

    def test_logcal(self):
        #check that logcal give 0 chi2 for all 20 testinfos
        diff = np.zeros(20)
        for index in range(20):
            arrayinfopath = os.path.dirname(os.path.realpath(__file__)) + '/testinfo/test'+str(index+1)+'_array_info.txt'
            calibrator = Oc.RedundantCalibrator(56)
            calibrator.compute_redundantinfo(arrayinfopath, tol=.1)
            if False: # XXX this was to migrate files so they include bl order w/ data
                _info = calibrator.Info # XXX needs to have been initialized the old way (w/o reds)
                datapath = os.path.dirname(os.path.realpath(__file__)) + '/testinfo/test'+str(index+1)+'_data.txt'
                with open(datapath) as f:
                    rawinfo = [[float(x) for x in line.split()] for line in f]
                data = np.array([i[0] + 1.0j*i[1] for i in rawinfo[:-1]],dtype = 'complex64') #last element is empty
                data = data.reshape((1,1,len(data)))
                dd = _info.make_dd(data)
                np.savez('calib_test_data_%02d.npz' % index, bls=np.array(dd.keys()), vis=np.array(dd.values()))
            info = calibrator.Info
            npz = np.load(testdata % index)
            bls = [tuple(bl) for bl in npz['bls']]
            dd = dict(zip(bls, npz['vis']))
            data = info.order_data(dd)
            ####do calibration################
            calibrator.removeDegeneracy = True
            calibrator.removeAdditive = False
            calibrator.keepData = True
            calibrator.keepCalpar = True
            calibrator.convergePercent = 1e-5
            calibrator.maxIteration = 50
            calibrator.stepSize = .2
            calibrator.computeUBLFit = True

            calibrator.logcal(data, np.zeros_like(data), verbose=VERBOSE)
            log = np.copy(calibrator.rawCalpar)
            ampcal = log[0,0,3:info['nAntenna']+3]
            phasecal = log[0,0,info['nAntenna']+3: info['nAntenna']*2+3]
            calpar = 10**(ampcal)*np.exp(1.0j*phasecal)
            start_ubl = 3 + 2*info['nAntenna']
            end_ubl = start_ubl + 2*len(info.ublcount)
            ublfit = log[0,0,start_ubl:end_ubl:2]+1.0j*log[0,0,start_ubl+1:end_ubl+1:2]
            ####import real calibration parameter
            calparpath = os.path.dirname(os.path.realpath(__file__)) + '/testinfo/test'+str(index+1)+'_calpar.txt'
            with open(calparpath) as f:
                rawinfo = [[float(x) for x in line.split()] for line in f]
            temp = np.array(rawinfo[:-1])
            correctcalpar = (np.array(temp[:,0]) + 1.0j*np.array(temp[:,1]))[info['subsetant']]
            ###compare calpar with correct calpar
            overallfactor = np.real(np.mean(ublfit))**0.5
            diffnorm = la.norm(calpar*overallfactor - correctcalpar)
            self.assertAlmostEqual(diffnorm, 0, 4)

    def test_lincal(self):
        fileindex = 3      #use the 3rd file to do the test, can also change this to any number from 1 to 20
        length = 100
        loglist = np.zeros(length)
        linlist = np.zeros(length)

        ####import arrayinfo################
        arrayinfopath = os.path.dirname(os.path.realpath(__file__)) + '/testinfo/test'+str(fileindex)+'_array_info.txt'
        nant = 56
        calibrator = Oc.RedundantCalibrator(nant)
        calibrator.compute_redundantinfo(arrayinfopath)
        info = calibrator.Info
        npz = np.load(testdata % (fileindex-1))
        bls = [tuple(bl) for bl in npz['bls']]
        dd = dict(zip(bls, npz['vis']))
        data = info.order_data(dd)

        ####Config parameters###################################
        needrawcal = True #if true, (generally true for raw data) you need to take care of having raw calibration parameters in float32 binary format freq x nant
        std = 0.1

        ####do calibration################
        calibrator.removeDegeneracy = True
        calibrator.removeAdditive = False
        calibrator.keepData = True
        calibrator.keepCalpar = True
        calibrator.convergePercent = 1e-5
        calibrator.maxIteration = 50
        calibrator.stepSize = .2
        calibrator.computeUBLFit = True

        for i in range(length):
            noise = (np.random.normal(scale = std, size = data.shape) + 1.0j*np.random.normal(scale = std, size = data.shape)).astype('complex64')
            ndata = data + noise
            calibrator.logcal(ndata, np.zeros_like(ndata), verbose=VERBOSE)
            calibrator.lincal(ndata, np.zeros_like(ndata), verbose=VERBOSE)

            linchi2 = (calibrator.rawCalpar[0,0,2]/(info['At'].shape[1] - info['At'].shape[0])/(2*std**2))**0.5
            logchi2 = (calibrator.rawCalpar[0,0,1]/(info['At'].shape[1] - info['At'].shape[0])/(2*std**2))**0.5
            linlist[i] = linchi2
            loglist[i] = logchi2

            # The sum of the chi^2's for all antennas should be twice the
            # calibration chi^2. Omnical uses single precision floats in C, so
            # the ratio of that sum to chi^2 must be equal to 2 to 5 decimal
            # places. That should hold in all cases.
            chi2all = calibrator.rawCalpar[0,0,2]
            chi2ant = calibrator.rawCalpar[0,0,3+2*(info.nAntenna + len(info.ublcount)):]
            np.testing.assert_almost_equal(np.sum(chi2ant)/chi2all-2, 0, 5)

        self.assertTrue(abs(np.mean(linlist)-1.0) < 0.01)        #check that chi2 of lincal is close enough to 1
        self.assertTrue(np.mean(linlist) < np.mean(loglist))     #chick that chi2 of lincal is smaller than chi2 of logcal

if __name__ == '__main__':
    unittest.main()
