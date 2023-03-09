# numpy
* 굉장히 큰 matrix 와 많은 수식
* python 은 인터프리터 언어 $\rightarrow$ 처리속도 문제

#### $\rightarrow$ numpy를 사용하자
## numpy(numerical python)
* 넘파이라고 부름
* 파이썬의 고성능 과학 계산용 패키지
* matrix 와 vector 같은 array 연산의 실질적 표준
* 이학 공학 금융학 등에서 많이 사용
  #### 특징
  * 일반 리스트에 비해 빠르고 메모리가 효율적이다
  * **반복문 없이 데이터 배열에 대한 처리를 지원한다**
  * 선형대수와 관련된 다양한 기능을 제공한다
  * c, c++ 등 여러 언어와 통합 가능하다

## ndarray
* numpy 에서 사용하는 배열
* np.array 함수를 사용해 생성
* 하나의 데이터 타입만 배열에 넣을 수 있음 $\rightarrow$ dynamic typing not supported 
* c언어 의 array 를 사용해 배열 생성
```python
first_array = np.array([1, 2, 3, 4], float)
first_array
# array([1., 2., 3., 4.])

type(first_array[1])
# numpy.float64
```
* 리스트와 달리 데이터가 차례대로 저장됨 $\rightarrow$ 연산 속도가 빨라짐, 메모리 공간 잡기도 효율적
![img](https://miro.medium.com/v2/resize:fit:720/format:webp/1*WPRfgLC-j3BYi_G1ntqCRA.png)
<div style="text-align: right"> [출처] https://python.plainenglish.io/exploring-python-data-structures-single-linked-lists-part-1-cb0f92cdb7d9 </div>

```python
a_list = [1, 2, 3]
b_list = [3, 2, 1]
a_list[0] is b_list[-1]
# True

a_list = np.array([1, 2, 3])
b_list = np.array([3, 2, 1])
a_list[0] is b_list[-1]
# False
```