# Logic_gate
#### 하나 이상의 논리적 입력값에 대해 논리 연산을 수행하여 하나의 논리적 출력 값을 얻는 전자회로
---
## 개념
* AND 게이트 : 모든 값이 참일 때 참을 출력
* OR 게이트 : 하나의 값이라도 참일 때 참을 출력
* XOR 게이트 : 하나의 값만 참일 때 참을 출력, 단일 선형 함수로 나타낼 수 없음
* 이론적으로는 NAND 게이트 만으로 컴퓨터도 만들 수 있음

## 목표
* 게이트의 구성과 다중 퍼셉트론의 단순 구현

## 코드
```python
# --------------------------- modules --------------------------- #

import numpy as np
import matplotlib.pyplot as plt

# -------------------------- functions -------------------------- #

# AND 게이트
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    cal = np.sum(w*x) + b
    if cal <= 0:
        return 0
    else:
        return 1

# OR 게이트
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    cal = np.sum(w*x) + b
    if cal <= 0:
        return 0
    else:
        return 1

# XOR 게이트
def XOR(x1, x2):
    s1 = 1 - AND(x1, x2) #NAND 게이트
    s2 = OR(x1, x2)
    return AND(s1, s2)

# -------------------------- progress -------------------------- #

print(f'AND(0, 0) => {AND(0, 0)}')
print(f'AND(0, 1) => {AND(0, 1)}')
print(f'AND(1, 0) => {AND(1, 0)}')
print(f'AND(1, 1) => {AND(1, 1)}')

print(f'OR(0, 0) => {OR(0, 0)}')
print(f'OR(0, 1) => {OR(0, 1)}')
print(f'OR(1, 0) => {OR(1, 0)}')
print(f'OR(1, 1) => {OR(1, 1)}')

# ---------------------------- plot ---------------------------- #
```
<img src="/images/xor_gate.png" style="margin-right: auto; margin-left: auto;">