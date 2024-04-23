import re


class HypothesisBuffer:
  SEP = ' '

  def __init__(self):
    self.new = []
    self.buffer = []
    self.trunk = []

  def __lcs(self, a, b):
    m, n = (len(x) for x in (a, b))
    L = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
      for j in range(n + 1):
        if i == 0 or j == 0:
          L[i][j] = 0
        elif len(set(re.sub(r'\W+', '', x).lower()
                     for x in (a[i - 1], b[j - 1]))) == 1:
          L[i][j] = L[i - 1][j - 1] + 1
        else:
          L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return L[m][n]

  def __call__(self, new: list):
    self.new = new.split(self.SEP)[self.__lcs(new, self.trunk):]
    cn = len(self.trunk)
    nn = len(self.new)
    for i in range(1, min(cn, nn, 5) + 1):
      c = self.SEP.join([str(self.trunk[-j]) for j in range(1, i + 1)][::-1])
      tail = self.SEP.join(str(self.new[j - 1]) for j in range(1, i + 1))
      if c == tail:
        for _ in range(i):
          self.new.pop(0)
        break
    return self.__write()

  def __write(self):
    commit = []
    while self.new and self.buffer:
      new_token, buffered_token = (
          getattr(self, x)[0]
          for x in ('new', 'buffer')
      )
      norm = [re.sub(r'\W+', '', x).lower()
              for x in (new_token, buffered_token)]
      try:
        for norm_token, x in zip(norm, ('new', 'buffer')):
          if len(norm_token) == 0:
            getattr(self, x).pop(0)
            raise Exception()
      except Exception:
        continue
      if len(set(norm)) != 1:
        break
      commit.append(new_token)
      for x in ('new', 'buffer'):
        getattr(self, x).pop(0)
    self.buffer = self.new
    self.trunk.extend(commit)
    return self.SEP.join(self.trunk)
