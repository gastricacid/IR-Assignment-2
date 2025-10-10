##  @file
#   @brief PAROL6 Robot defined by standard DH parameters with 3D model
#   @date    2025-10-06
#
#   Note: link lengths (mm -> m) are placed directly into the RevoluteDH calls
#         and into the qtest_transforms to match the style of your UR3 sample.
#

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os
from math import pi

class PAROL6(DHRobot3D):
    def __init__(self):
        """
        PAROL6 Robot by DHRobot3D class (standard DH)
        Standby pose (actual angles): [0, -90°, 180°, 0, 0, 180°] mapped to q = [0..0]
        """
        links = self._create_DH()

        link3D_names = dict(
            link0 = 'base_parol6',
            link1 = 'shoulder_parol6',
            link2 = 'upperarm_parol6',
            link3 = 'forearm_parol6',
            link4 = 'wrist1_parol6',
            link5 = 'wrist2_parol6',
            link6 = 'wrist3_parol6'
        )

        # qtest = zeros because offsets make standby pose = q = 0
        qtest = [0, 0, 0, 0, 0, 0]

        # Inline numeric values (mm->m) used directly here to match UR3 style.
        # These transforms are approximate; tune to align your 3D models exactly.
        qtest_transforms = [
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0.11050) @ spb.trotz(pi),
            spb.transl(0, -0.05, 0.11050 + 0.02) @ spb.trotz(pi),
            spb.transl(0, -0.02, 0.11050 + 0.20) @ spb.trotz(pi),
            spb.transl(0, -0.02, 0.11050 + 0.35) @ spb.rpy2tr(0, -pi/2, pi),
            spb.transl(0.0, -0.09, 0.11050 + 0.35) @ spb.rpy2tr(0, -pi/2, pi),
            spb.transl(-0.08, -0.09, 0.11050 + 0.36) @ spb.trotz(pi)
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name='PAROL6', link3d_dir=current_path,
                         qtest=qtest, qtest_transforms=qtest_transforms)

        self.q = qtest

    def _create_DH(self):

        a = [0.02342, 0.18, -0.0435, 0.0, 0.0, -0.04525]
        d = [0.11050, 0.0, 0.0, -0.17635, 0.0, -0.0628]
        alpha = [-pi/2, pi, pi/2, -pi/2, pi/2, pi]
        offsets = [0.0, -pi/2, pi, 0.0, 0.0, pi]
        qlim = [[-2*pi, 2*pi] for _ in range(6)]

        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offsets[i], qlim=qlim[i])
            links.append(link)
        return links

    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        self.q = self._qtest
        self.base = SE3(0.5, 0.5, 0)
        self.add_to_env(env)

        q_goal = [self.q[i] - pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        for q in qtraj:
            self.q = q
            env.step(0.02)
        time.sleep(3)


if __name__ == "__main__":
    r = PAROL6()
    r.test()
