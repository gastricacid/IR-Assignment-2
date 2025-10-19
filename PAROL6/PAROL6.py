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
        PAROL6 Robot on a Linear Rail by DHRobot3D class (standard DH)
        First joint is prismatic (rail), followed by 6 revolute joints
        Standby pose: prismatic at 0, revolute joints at their offset positions (q = [0,0,0,0,0,0,0])
        """
        links = self._create_DH()

        link3D_names = dict(
            link0 = 'base_rail',
            link1 = 'slider_rail',
            link2 = 'shoulder_parol6',
            link3 = 'upperarm_parol6',
            link4 = 'forearm_parol6',
            link5 = 'wrist1_parol6',
            link6 = 'wrist2_parol6',
            link7 = 'wrist3_parol6',
            color0 = (0.2, 0.2, 0.2, 1),
            color1 = (0.1, 0.1, 0.1, 1)
        )

        # qtest = zeros: prismatic at 0, revolute offsets make standby pose = q = 0
        qtest = [0, 0, 0, 0, 0, 0, 0]

        # Inline numeric values (mm->m) used directly here to match UR3 style.
        # Rail transforms first, then PAROL6 links mounted on slider
        qtest_transforms = [
            spb.transl(0, 0, 0),                                                    # base_rail
            spb.trotx(-pi/2),                                                       # slider_rail
            spb.transl(0, 0, 0.11050) @ spb.trotz(pi),                             # shoulder (on slider)
            spb.transl(0, -0.05, 0.11050 + 0.02) @ spb.trotz(pi),                  # upperarm
            spb.transl(0, -0.02, 0.11050 + 0.20) @ spb.trotz(pi),                  # forearm
            spb.transl(0, -0.02, 0.11050 + 0.35) @ spb.rpy2tr(0, -pi/2, pi),       # wrist1
            spb.transl(0.0, -0.09, 0.11050 + 0.35) @ spb.rpy2tr(0, -pi/2, pi),     # wrist2
            spb.transl(-0.08, -0.09, 0.11050 + 0.36) @ spb.trotz(pi)               # wrist3
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name='PAROL6', link3d_dir=current_path,
                         qtest=qtest, qtest_transforms=qtest_transforms)

        self.q = qtest

    def _create_DH(self):
        # Start with prismatic link (rail)
        links = [rtb.PrismaticDH(theta=pi, a=0, alpha=pi/2, qlim=[-0.8, 0])]
        
        # PAROL6 revolute links
        a = [0.02342, 0.18, -0.0435, 0.0, 0.0, -0.04525]
        d = [0.11050, 0.0, 0.0, -0.17635, 0.0, -0.0628]
        alpha = [-pi/2, pi, pi/2, -pi/2, pi/2, pi]
        offsets = [0.0, -pi/2, pi, 0.0, 0.0, pi]
        qlim = [[-2*pi, 2*pi] for _ in range(6)]

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
        q_goal[0] = -0.8  # Move the rail link
        qtraj = rtb.jtraj(self.q, q_goal, 50).q
        for q in qtraj:
            self.q = q
            env.step(0.02)
        env.hold()
        time.sleep(3)


if __name__ == "__main__":
    r = PAROL6()
    r.test()
