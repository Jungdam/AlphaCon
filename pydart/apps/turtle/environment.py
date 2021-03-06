from abc import ABCMeta, abstractmethod
import time
import multiprocessing as mp


class EnvironmentBase:
    __metaclass__ = ABCMeta

    def __init__(self):
        return

    @abstractmethod
    def reset(self):
        """Reset environment."""
        raise NotImplementedError("Must override")

    @abstractmethod
    def state(self):
        """Returns current state vector
        Returns:
        state -- a current state vector (numpy array)
        """
        raise NotImplementedError("Must override")

    @abstractmethod
    def step(self, action):
        """Returns current state vector
        Keyword arguments:
        action -- action parameter for stepping forward
        Returns:
        reward -- a reward value after one step (float)        
        """
        raise NotImplementedError("Must override")

    @abstractmethod
    def render(self):
        """Render the current environment"""
        raise NotImplementedError("Must override")


class Environment_Slave(mp.Process):
    def __init__(self, idx, q_input, q_result, func_gen_env, args_gen_env):
        super(Environment_Slave, self).__init__()
        self.env = func_gen_env(args_gen_env)
        self.response_time = 0.001
        self.q_input = q_input
        self.q_result = q_result
        self.idx = idx

    def error(self, msg):
        print "[Environment_Slave] error -", self.proc_name, idx, msg

    def run(self):
        self.proc_name = mp.current_process().name
        print self.proc_name, self.idx, 'Hello!'
        while True:
            if self.q_input.empty():
                time.sleep(self.response_time)
                continue
            signal, data = self.q_input.get()
            if signal == "step":
                self.reward = self.env.step(data)
                self.q_result.put(self.reward)
            elif signal == "state":
                self.q_result.put(self.env.state())
            elif signal == "reset":
                self.env.reset()
            elif signal == "terminate":
                return
            else:
                self.run_extra(signal, data)

    def run_extra(self, signal, data):
        return


class Environment_Master:
    def __init__(self, num_slave, func_gen_env, args_gen_env, env_slave_custom=None):
        self.num_slave = num_slave
        self.q_inputs = []
        self.q_results = []
        self.slaves = []
        for i in range(num_slave):
            q_input = mp.Queue()
            q_result = mp.Queue()
            self.q_inputs.append(q_input)
            self.q_results.append(q_result)
            if env_slave_custom is None:
                slave = Environment_Slave(i, q_input, q_result, func_gen_env, args_gen_env)
            else:
                slave = env_slave_custom(i, q_input, q_result, func_gen_env, args_gen_env)
            self.slaves.append(slave)
            slave.start()

    def __del__(self):
        self.run(self.num_slave * ["terminate"])

    def check_empty_result(self):
        for q in self.q_results:
            if not q.empty():
                print "[Environment_Master] error - q is not empty"
                return False
        return True

    def run(self, signals, data_in=None, return_data=True):
        if not self.check_empty_result():
            print 'Result queue is not empty... something wrong...'
            return None
        for i in range(self.num_slave):
            if data_in is not None:
                self.q_inputs[i].put([signals[i], data_in[i]])
            else:
                self.q_inputs[i].put([signals[i], None])
        if return_data:
            result = []
            for q in self.q_results:
                while q.empty():
                    continue
                result.append(q.get())
            return result
        return None

    def state(self):
        return self.run(self.num_slave * ["state"])

    def reward(self):
        return self.run(self.num_slave * ["reward"])

    def step(self, wingbeats):
        return self.run(self.num_slave * ["step"], wingbeats)

    def reset(self, idx=None):
        if idx is not None:
            self.q_inputs[idx].put(["reset", None])
        else:
            self.run(self.num_slave * ["reset"], return_data=False)
