from gymnasium.wrappers import RecordEpisodeStatistics
import csv
import os

from gymnasium.wrappers import RecordEpisodeStatistics
import csv
import os

from RL.common.utils.decorator.Meta import SerializeMeta


class CSVLoggerWrapper(RecordEpisodeStatistics,metaclass=SerializeMeta):
    """
    CSVLoggerWrapper 继承自 RecordEpisodeStatistics，
    在每个 episode 结束时将 `r`、`l`、`t` 及自定义 info_keywords 写入 CSV。
    如果 filename 为空或 None，则仅保留 RecordEpisodeStatistics 功能，不写文件。
    """
    __exclude__=["_file","_writer"]

    def __deserialize_post__(self):
        self._init(self.filename,self.info_keywords)
    def _init(self,filename,info_keywords=()):
        # 仅当给定有效 filename 时，初始化 CSV 写入
        if filename:
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            # 打开文件并写入表头
            self._file = open(filename, 'w', newline='')
            headers = ['episode', 'r', 'l', 't'] + list(info_keywords)
            self._writer = csv.writer(self._file)
            self._writer.writerow(headers)
    def __init__(self, env, filename: str = None, info_keywords=()):
        # 初始化父类，记录 episode 统计到 info['episode']
        super().__init__(env)
        self.info_keywords = info_keywords
        self._file = None
        self._writer = None
        self.episode_count = 0
        self.filename = filename
        self._init(filename,info_keywords)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        # 当 episode 结束且启用了文件写入时，将统计写入 CSV
        if done and 'episode' in info and self._writer:
            self.episode_count += 1
            ep = info['episode']
            row = [self.episode_count,ep.get('r'), ep.get('l'), ep.get('t')]
            for k in self.info_keywords:
                row.append(info.get(k, ''))
            self._writer.writerow(row)
            self._file.flush()
        return obs, reward, terminated, truncated, info