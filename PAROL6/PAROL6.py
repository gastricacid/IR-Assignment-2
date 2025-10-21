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
            link2 = 'shoulder',
            link3 = 'upperarm',
            link4 = 'forearm',
            link5 = 'wrist1',
            link6 = 'wrist2',
            link7 = 'wrist3',
            color0 = [0.2, 0.2, 0.2],      # base_rail - dark gray
            color1 = [0.3, 0.3, 0.3],      # slider_rail - medium gray
            color2 = [0.8, 0.3, 0.2],      # shoulder - orange/red
            color3 = [0.2, 0.5, 0.8],      # upperarm - blue
            color4 = [0.3, 0.7, 0.3],      # forearm - green
            color5 = [0.9, 0.7, 0.2],      # wrist1 - yellow/gold
            color6 = [0.7, 0.3, 0.7],      # wrist2 - purple
            color7 = [0.9, 0.5, 0.2]       # wrist3 - orange
        )

        # qtest = zeros: prismatic at 0, revolute offsets make standby pose = q = 0
        qtest = [0, 0, 0, 0, 0, 0, 0]

        # Inline numeric values (mm->m) used directly here to match UR3 style.
        # Rail transforms first, then PAROL6 links mounted on slider
        # After trotx(pi/2), z becomes y, and y becomes -z in the rotated frame
        qtest_transforms = [
            spb.transl(0, 0, 0) @ spb.trotx(pi/2),                                  # base_rail (rotated 90° from original)
            spb.trotx(-pi/2) @ spb.trotx(pi/2),                                     # slider_rail (original + 90° in x-axis)
            spb.transl(0.5, -0.38050, 2.4) @ spb.trotx(0) @ spb.trotz(pi),           # shoulder (on slider) - z->y after rotation
            spb.transl(0, 0.11050 + 0.02, -0.05) @ spb.trotx(pi/2) @ spb.trotz(pi), # upperarm - y->-z after rotation
            spb.transl(0, 0.11050 + 0.20, 3) @ spb.troty(-pi/14) @ spb.troty(pi) @ spb.trotz(0), # forearm
            spb.transl(-0.2, 0.11050 + 0.35, 2.5) @ spb.trotx(pi/2) @ spb.rpy2tr(pi/2, -pi/2, pi), # wrist1
            spb.transl(-1.05, 2.1050, 2.1) @ spb.troty(-pi/2) @ spb.trotx(pi/2) @ spb.troty(pi/20) @ spb.rpy2tr(0, -pi/2, pi), # wrist2
            spb.transl(-0.08, 0.11050 + 0.36, -0.09) @ spb.trotx(pi/2) @ spb.trotz(pi) # wrist3
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
        import random
        env = swift.Swift()
        env.launch(realtime=True)
        self.q = self._qtest
        self.base = SE3(0.5, 0.5, 0)
        self.add_to_env(env)

        # First target joint positions: [rail, j1, j2, j3, j4, j5, j6]
        from math import radians
        q_goal1 = [
            0,              # Prismatic rail at 0
            radians(0),     # Joint 1: 0°
            radians(-90),   # Joint 2: -90°
            radians(180),   # Joint 3: 180°
            radians(0),     # Joint 4: 0°
            radians(0),     # Joint 5: 0°
            radians(180)    # Joint 6: 180°
        ]
        qtraj1 = rtb.jtraj(self.q, q_goal1, 50).q
        for q in qtraj1:
            self.q = q
            env.step(0.02)

        # Pause before moving to the next trajectory
        time.sleep(1)


        # Joint limits (in radians or meters):
        # 0: prismatic [0, 0.8]
        # 1: revolute [-250°, 250°]
        # 2: revolute [-141°, 141°]
        # 3: revolute [-180°, 180°]
        # 4: revolute [-212°, 212°]
        # 5: revolute [-180°, 180°]
        # 6: revolute [-inf, inf] (use [-2*pi, 2*pi] for practical demo)
        deg = lambda x: x * pi / 180
        q_rand = [random.uniform(0, 0.8),
                  random.uniform(-deg(250), deg(250)),
                  random.uniform(-deg(141), deg(141)),
                  random.uniform(-deg(180), deg(180)),
                  random.uniform(-deg(212), deg(212)),
                  random.uniform(-deg(180), deg(180)),
                  random.uniform(-2*pi, 2*pi)]

        print("Moving to random joint configuration:", q_rand)
        qtraj2 = rtb.jtraj(self.q, q_rand, 50).q
        for q in qtraj2:
            self.q = q
            env.step(0.02)

        env.hold()
        time.sleep(3)


if __name__ == "__main__":
    r = PAROL6()
    r.test()
