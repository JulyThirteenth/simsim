##########################################
# author: ShawWang
# email: shawwang@yeah.net
# github: JulyThirteenth
# personal pages: julythirteenth.github.io
##########################################

import numpy as np


class Robot:
    def __init__(self, state: np.array, dt: float) -> None:
        self.state = state
        self.dt = dt

    def set_state(self, state: np.array):
        self.state = state

    def step(self, control: np.array, method: str = 'forward-euler'):
        self.state[-2:] = control
        if method == 'forward-euler':
            self.state = self.forward_euler()
        elif method == 'runge-kutta':
            self.state = self.runge_kutta()
        else:
            raise ValueError(f'Unknown method {method}.')
        return self.state

    def step_with_noise(self, control: np.array, method: str = 'forward-euler'):
        noise = np.random.normal(0, 1e-3, control.shape)
        return self.step(control + noise, method)

    '''
    @brief Forward Euler method
    '''

    def forward_euler(self):
        return self.state + self.kinematic(self.state) * self.dt

    '''
    @brief Runge-Kutta method
    '''

    def runge_kutta(self):
        dx1 = self.kinematic(self.state)
        dx2 = self.kinematic(self.state + 0.5 * self.dt * dx1)
        dx3 = self.kinematic(self.state + 0.5 * self.dt * dx2)
        dx4 = self.kinematic(self.state + self.dt * dx3)
        return self.state + self.dt * (dx1 + 2 * dx2 + 2 * dx3 + dx4) / 6.0

    def kinematic(self, state: np.array):
        raise NotImplementedError


class Diff(Robot):
    def __init__(self, state: np.array, dt: float):
        super().__init__(state, dt)

    '''
    @param state: [x, y, theta, v, omega]
    '''

    def kinematic(self, state: np.array):
        return np.array([state[3] * np.cos(state[2]),
                         state[3] * np.sin(state[2]),
                         state[4], 0.0, 0.0])


class Car(Robot):
    def __init__(self, state: np.array, dt: float, max_kappa: float):
        super().__init__(state, dt)
        self.max_kappa = max_kappa
        self.width = 2.0
        self.rear_hang = 1.0
        self.wheelbase = 2.0
        self.front_hang = 1.0
        self.length = self.rear_hang + self.wheelbase + self.front_hang

    def get_upper_left_vertex(self):
        vertex = np.array([self.length / 2.0, self.width / 2.0])
        vertex += np.array([(self.front_hang + self.wheelbase - self.rear_hang) / 2.0, 0.0])
        return Car.rotation_matrix_2d(self.state[2]) @ vertex + self.state[:2]

    def get_upper_right_vertex(self):
        vertex = np.array([self.length / 2.0, -self.width / 2.0])
        vertex += np.array([(self.front_hang + self.wheelbase - self.rear_hang) / 2.0, 0.0])
        return Car.rotation_matrix_2d(self.state[2]) @ vertex + self.state[:2]

    def get_bottom_left_vertex(self):
        vertex = np.array([-self.length / 2.0, self.width / 2.0])
        vertex += np.array([(self.front_hang + self.wheelbase - self.rear_hang) / 2.0, 0.0])
        return Car.rotation_matrix_2d(self.state[2]) @ vertex + self.state[:2]

    def get_bottom_right_vertex(self):
        vertex = np.array([-self.length / 2.0, -self.width / 2.0])
        vertex += np.array([(self.front_hang + self.wheelbase - self.rear_hang) / 2.0, 0.0])
        return Car.rotation_matrix_2d(self.state[2]) @ vertex + self.state[:2]

    @staticmethod
    def rotation_matrix_2d(theta):
        """返回二维旋转矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    '''
    @param state: [x, y, theta, v, kappa]
    '''

    def kinematic(self, state: np.array):
        self.state[4] = np.clip(self.state[4], -self.max_kappa, self.max_kappa)
        return np.array([state[3] * np.cos(state[2]),
                         state[3] * np.sin(state[2]),
                         state[3] * state[4], 0.0, 0.0])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # state = np.array([0., 0., 0., 0., 0.])
    # dt = 0.1
    # robot = Omni(state, dt)
    # step = 100
    # v = np.linspace(0.0, 1.0, step)
    # omega = np.linspace(0.0, np.pi, step)
    # control = np.vstack([v, omega])
    # states = [state]
    # for i in range(step):
    #     state = robot.step(control[:, i])
    #     states.append(state)
    # plt.subplot(121)
    # plt.plot(np.array(states)[:, 0], np.array(states)[:, 1], c='r')
    # state = np.array([0., 0., 0., 0., 0.])
    # dt = 0.1
    # robot = Car(state, dt, 0.2)
    # step = 10
    # v = np.linspace(0.0, 1.0, step)
    # kappa = np.linspace(0.0, 0.2, step)
    # control = np.vstack([v, kappa])
    # states = [state]
    # for i in range(step):
    #     state = robot.step(control[:, i])
    #     states.append(state)
    # plt.subplot(122)
    # plt.plot(np.array(states)[:, 0], np.array(states)[:, 1], c='b')
    # plt.show()

    from matplotlib.patches import Circle, Rectangle
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-20.0, 20.0)
    ax.set_ylim(-20.0, 20.0)
    ax.set_aspect('equal')
    state = np.array([0., 0., np.pi / 2.0, 0., 0.])
    dt = 0.1

    robot = Diff(state, dt)
    step = 100
    v = np.linspace(0.0, 10.0, step)
    omega = np.linspace(0.0, -np.pi, step)
    control = np.vstack([v, omega]).transpose()
    footprint = None


    def update(frame):
        global footprint
        state = robot.step(control[frame], 'runge-kutta')
        if footprint is not None:
            footprint.remove()
        footprint = Circle((state[0], state[1]), radius=0.5, color='r')
        ax.add_patch(footprint)
        return ax


    # robot = Car(state, dt, 0.2)
    # step = 100
    # v = np.linspace(2.0, 2.0, step)
    # kappa = np.linspace(-0.2, 0.2, step)
    # control = np.vstack([v, kappa]).transpose()
    # footprint = None

    # def update(frame):
    #     global footprint
    #     state = robot.step(control[frame], 'runge-kutta')
    #     if footprint is not None:
    #         footprint.remove()
    #     footprint = Rectangle(robot.get_bottom_left_vertex(),
    #                           robot.width, robot.length,
    #                           angle=np.rad2deg(state[2]) - 90.0)
    #     ax.add_patch(footprint)
    #     return ax

    ani = FuncAnimation(fig, update, frames=len(control), interval=dt)
    ani.save('motion.gif', fps=25, writer='pillow')
