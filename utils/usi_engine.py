import asyncio
import sys
from enum import Enum, unique
import os
import re
import pandas as pd
from scipy import stats
import numpy as np

@unique
class State(Enum):
    NOT_READY = 1
    READY = 2
    WORKING = 3

class USIEngine:
    count = 0
    done_bestmove = 0
    engines = []
    # pvre = re.compile(r'.*multipv (\d+) .* score cp (-?\d+) .* pv ([\w\+]+)')
    re_multi = re.compile(r'.*multipv (\d+)')
    re_score = re.compile(r'.* score (\w+) (-?\d+) .*')
    re_mate = re.compile(r'.* score mate (-?\d+) .*')
    re_depth = re.compile(r'.* depth (-?\d+) .*')
    re_pv = re.compile(r'.* pv ([\w\+\*]+)')
    ply = 0

    @classmethod
    async def create(cls, proc):
        e = USIEngine()
        e.index = USIEngine.count
        e.proc = proc
        e.multi_pv = 1
        # await e.set_option("MultiPV", e.multi_pv)
        # await e.set_option("NetworkDelay", 2)
        # await e.set_option("NetworkDelay2", 2)
        # await e.set_option("Threads", 8)
        e.pvs = [None] * e.multi_pv
        e.output = ""
        e.watch_task = asyncio.create_task(e.watcher())
        e.state = State.NOT_READY
        
        USIEngine.engines.append(e)
        USIEngine.count += 1
        return e

    @classmethod
    async def wait_all(cls, answer, callback=None):
        await asyncio.gather(*[e.wait_for(answer) for e in USIEngine.engines])
        if callback is not None:
            await callback()

    @classmethod
    async def send_all(cls, command, callback=None):
        USIEngine.done_bestmove = 0
        await asyncio.gather(*[e.send(command) for e in USIEngine.engines])
        if callback is not None:
            await callback()

    @classmethod
    def agg_bestmove(cls):
        print("agg_bestmove()", flush=True)
        print(f"USIEngine.done_bestmove: {USIEngine.done_bestmove}", flush=True)
        if USIEngine.done_bestmove < len(USIEngine.engines):
            return

        for e in USIEngine.engines:
            if "resign" in e.best_move:
                print(f"e.best_move: {e.best_move}", flush=True)
                print("# info resign found!", flush=True)
                print(e.best_move, flush=True)
                return

        pvs = []

        df = pd.DataFrame(columns=['move',])
        for ei, e in enumerate(USIEngine.engines):
            print(f"e.pvs: {e.pvs}", flush=True)
            filtered = list(filter(lambda pv: pv is not None, e.pvs))
            print(f"filtered: {filtered}", flush=True)
            if len(filtered) == 0:
                break
            pvs.append(filtered)
            d = pd.DataFrame(filtered, columns=['move', ei])
            # d[ei] = d[ei].apply(USIEngine.sigmoid)
            print(d, flush=True)
            df = df.merge(d, on=['move'], how='outer')

        print(df, flush=True)
        min_value = df.min(axis=0)
        df = df.fillna(min_value)
        df['gmean'] = stats.gmean(df[list(range(USIEngine.count))], axis=1)
        df['amean'] = np.mean(df[list(range(USIEngine.count))], axis=1)
        df['amean'] = df['amean'].apply(USIEngine.sigmoid)
        df = df.sort_values(by='amean', ascending=False, )
        best_move = df['move'].iloc[0]
        # print(df, flush=True)

        # Ply-based
        print(f"USIEngine.ply: {USIEngine.ply}", flush=True)
        if (USIEngine.ply <= 60):
            col = 0
        else:
            col = 1
        print(f"col: {col}", flush=True)
        df = df.sort_values(by=col, ascending=False)
        best_move = df['move'].iloc[0]

        print(f"bestmove {best_move}", flush=True)

    @classmethod
    def sigmoid(cls, x, d = 600):
        return 1 / (1 + np.exp(-x / d))

    async def set_option(self, name, value):
        await self.send(f"setoption name {name} value {value}")

    async def clear_output(self):
        self.output = ""
        self.pvs = [None] * self.multi_pv

    async def watcher(self):
        while True:
            data = await self.proc.stdout.readline()
            line = data.decode('ascii').strip()
#             if line != "":
#                 print(f"# info [engine{self.index}] {line}", flush=True)
            self.output += line

            if line.startswith('bestmove'):
#                 print("# info bestmove found!", flush=True)
                USIEngine.done_bestmove += 1
                self.best_move = line
#                 USIEngine.agg_bestmove()
            elif line.startswith('info '):
                m_depth = USIEngine.re_depth.match(line)
                m_score = USIEngine.re_score.match(line)
                # if (m_multi is not None) and (int(m_depth[1]) >= 2):
#                 print(f"m_score: {m_score}", flush=True)
                if (m_score is not None) and (int(m_score[2]) > -30000):
                    m_multi = USIEngine.re_multi.match(line)
#                     print(f"m_multi: {m_multi}", flush=True)
                    if m_multi is not None:
                        ipv = int(m_multi[1]) - 1
                    else:
                        ipv = 0
#                     print(f"ipv: {ipv}")
                    m_pv = USIEngine.re_pv.match(line)
#                     print(f"m_pv: {m_pv}", flush=True)
                    if m_score[1] == 'cp':
                        self.pvs[ipv] = [m_pv[1], int(m_score[2])]
                    else:
                        m_mate = USIEngine.re_mate.match(line)
                        sign = np.sign(int(m_score[2]))
                        self.pvs[ipv] = [m_pv[1], sign * 300000]

            elif line.startswith('id name'):
                line = "id name ensemble_cobra"

            elif line.startswith('quit'):
                self.watch_task.stop()
                self.watch_task.close()

#             if ((self.index == 0) and not line.startswith('bestmove')):
#                 print(line, flush=True)

    async def send(self, command):
        self.pvs = [None] * self.multi_pv
        self.line = ""

        if command[-1] != '\n':
            command += "\n"

        # com = command.split()[0]
        # if (com in ['go']):
        #     self.pvs = [None] * self.multi_pv
        #     self.line = ""

        self.proc.stdin.write(command.encode())
#         print(f"# info [>engine{self.index}] {command.rstrip()}", flush=True)

    async def wait_for(self, answer):
        while True:
            if answer in self.output:
                self.state = State.READY
                out = self.output
                self.output = ""
                return out

            await asyncio.sleep(0.1)

    async def send_wait(self, command, answer):
        self.send(command)
        self.wait_for(answer)

async def main(): 

    engines = []
    
    paths = [
        '/Users/hmats/workspace/DeepLearningShogi/x64/Release/dlshogi_tensorrt.exe',
        '/Users/hmats/Downloads/YaneuraOu-v5.33-windows/windows/NNUE/YaneuraOu_NNUE-tournament-clang++-zen2.exe'
    ]

    if sys.platform.startswith('linux'):
        paths = ['/mnt/c' + p for p in paths]

    # path must be absolute
    for path in (paths * 1):
        os.chdir(os.path.dirname(path))
        proc = await asyncio.create_subprocess_exec("./" + os.path.basename(path), stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE)
        # proc = await asyncio.create_subprocess_shell("./" + os.path.basename(path), stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE)
        engine = await USIEngine.create(proc)
        engines.append(engine)

    # await queue.put('usi')
    # await USIEngine.wait_all('usiok')

    # await queue.put('isready')
    # await USIEngine.wait_all('readyok')

    # Prepare stdin
    reader = asyncio.StreamReader()
    pipe = sys.stdin
    loop = asyncio.get_running_loop()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), pipe)

    async for line in reader:
        command = line.decode('ascii')
        # await queue.put(command)
        await USIEngine.send_all(command)
        if command.startswith('quit'):
            print("quitting...", flush=True)
            await asyncio.sleep(1)
            # loop = asyncio.get_running_loop()
            # loop.close()
            loop.call_soon_threadsafe(loop.stop)
            loop.call_soon_threadsafe(loop.close)
            sys.exit(0)
        elif command.startswith('position'):
            USIEngine.ply = len(command.split(' ')) - 3


    #await asyncio.sleep(3)

# asyncio.run(main())
