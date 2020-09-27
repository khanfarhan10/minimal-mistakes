---
title: "Recursion"
excerpt: "컴퓨터 과학에서의 재귀(Recursion)가 무엇인지 알아본다."
toc: true
toc_sticky: true
# toc_label: "페이지 주요 목차"

categories:
  - Algorithm

tags:
  - Recursive
  - Computer science
  - Mathematics

last_modified_at: 2019-07-10T08:06:00-05:00

header:
  overlay_image: /assets/images/headerIMG.jpg
  overlay_filter: 0.2 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
  #actions:
  #  - label: "More Info"
  #    url: "https://unsplash.com"

use_math: true
---

# Recursion

컴퓨터 과학에서의 **Recursion**은 어떤 문제가 그 문제의 **더 작은 사례의 해결**에 의존해 해결되는 방법론이다. 재귀적인 문제는 반복적인 방법으로도 풀어질 수 있는데 그렇게 되기 위해선 런 타임 이전에 더 작은 사례를 식별하고 인덱스화 해야한다. 컴퓨터 과학 그리고 프로그래밍에서 Recursion은 함수가 자기 코드(own code) 안에서 **자기 자신**을 호출함으로 재귀적인 문제를 해결한다. Recursion은 많은 종류의 문제에서 적용될 수 있고 컴퓨터 과학에서의 중요한 이론이므로 잘 알아둘 필요가 있다.

대부분의 프로그래밍 언어는 함수가 자기 코드(own code) 안에서 **자기 자신**을 호출하는 방식으로 재귀를 지원한다. 함수적 프로그래밍 언어는 반복적인 구조를 정의하지 않으므로 오직 재귀적인 방법에만 의존해 문제를 해결한다.

함수안에서 자기 자신을 반복적으로 호출하는 재귀적인 방식을 사용하면 콜스택의 크기가 계속 커져 모든 호출과 연관된 입력 크기의 합과 같아질 수 있고, 이는 반복적인 방법으로 쉽게 해결이 가능한 문제의 경우 재귀를 적용했을 때의 효율성이 떨어진다는 것을 말한다. 또한 재귀적인 방법을 적용할 때 다소 큰 문제에 대해서는 tail call 같은 최적화 기술을 사용하는 것이 필수적이다.

## Recursive functions

컴퓨터 과학에서 어떤 문제를 일반적인 방법 중의 하나는 **분할 정복(Divide and conquer)**이라는 것을 사용하는 것이다. 분할 정복은 어떤 문제를 같은 타입의 **더 작은 문제들로 분할**해 이를 해결하고 다시 합쳐 원래의 문제를 해결하는 방법론을 말한다. 여기서, **재귀 함수(Recursive functions)**를 이용하면 분할 정복을 보다 쉽게 적용할 수 있게된다. 문제에 따라서는 해당 문제의 더 작은 사례들이 겹칠 수가 있어 같은 문제를 반복적으로 풀지않고, 계산 시간을 절약하기 위해 더 작은 문제들의 솔루션을 저장한 **동적표 또는 룩업 테이블**(Dynamic table or look up table)을 구성하는데, 이를 함께 적용한 방법을 이를 **동적 프로그래밍 또는 메모이제이션(dynamic programming or memoization)**이라 부른다.

일반적으로 **재귀 함수(Recursive functions)**는 단순히 조건에 따라 값을 반환하는 하나 이상의 **기본 또는 기저 사례들(base cases)**과 자기 자신을 호출하는 하나 이상의 **재귀 사례들(Recursive cases)**로 구성된다. 팩토리얼을 계산을 예로 들면, 입력 값 n이 0 일때는 `0! = 1`의 단순히 1을 반환하는 기저 사례, 입력 값 n이 0 보다 클 때는 `n! = n(n − 1)!`의 재귀 사례로 구성할 수 있다.

```c++
unsigned int factorial(unsigned int n) {
    if (n == 0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}
```

재귀 함수에서의 재귀 사례는 보다 큰 입력을 단순한 입력으로 분해하는 것으로 볼 수 있다. 잘 설계된 재귀 함수에서의 각 재귀 호출은 입력 문제를 보다 단순한 사례로 분해하고 결국에는 기본 사례에 도달하게 한다. 주의할 점은 기본 사례를 **잘못 만들거나 생략하면(missing)** 재귀 함수는 **무한 루프(infinity loop)**에 빠질 수 있다.

## Single recursion and multiple recursion

재귀 함수가 단순히 **하나의 자기 호출(single self-reference)**을 가지고 있다면 이것을 **단일 재귀(Single recursion)**, **여러개의 자기 호출(multiple self-references)**을 가지고 있다면 **다중 재귀(multiple recursion)**라고 한다. 단일 재귀를 사용한 예로는 선형 탐색의 리스트 순회, 팩토리얼 계산이 있다. 이것들은 모두 하나의 재귀 사례로 행해질 수 있는 계산들이다.

**다음은 단일 재귀의 리스트 순회 코드이다.**

```java
public static void listTraversal(listNode node)
{
	// 현재 노드가 null이면 리턴
	if (node == null) {
	  return;
	}
	// 현재 노드의 데이터를 출력한다.
	System.out.print(root.data + " ");
	// 재귀적으로 현재 노드의 다음 노드를 탐색한다.
	listTraversal(node.next);
}
```

현재 노드가 null인지 체크해 재귀를 끝내는 기본 사례와 다음 노드로의 탐색을 진행하는 단일 재귀 호출로 구성되어 있음을 볼 수 있다.

---

**다중 재귀**를 사용한 예로는 깊이우선탐색 방식의 트리 순회를 들 수 있다. 대표적인 깊이우선탐색 방식의 트리 순회는 전위 순회, 중위 순회, 후위순회가 있으며 이들 모두는 2개의 다중 재귀 사례를 가지고 있다.

**다음은 트리 순회의 한 가지인 중위 순회의 코드이다.**

inorder 함수는 1개의 기본 사례와 2개의 재귀 사례로 구성되어 있음을 볼 수 있다.

```java
// Recursive function to perform in-order traversal of the tree
public static void inorder(TreeNode root)
{
	// 현재 노드가 null이면 리턴
	if (root == null) {
		return;
	}
	// 재귀적으로 왼쪽 서브트리를 탐색한다.
	inorder(root.left);
	// 현재 노드의 데이터를 출력한다. (방문)
	System.out.print(root.data + " ");
	// 재귀적으로 오른쪽 서브트리를 탐색한다.
	inorder(root.right);
}
```

단일 재귀는 대체로 다중 재귀보다 효율적이고, 일반적으로 선형 시간에 동작하고 상수적 메모리 공간을 요구하는 반복적인 계산 방법으로 대체될 수 있다. 그에 반해 다중 재귀는 지수적 시간 복잡도와 공간을 요구하고 명시적인 스택 없이는 반복적인 방법으로 대체될 수 없다.

다중 재귀나 반복적인 방법은 문제에 따라서 단일 재귀로 바뀔 수 있는데 피보나치 순열이 그 예이다. 피보나치 순열을 이전의 값들을 두 개의 다중 재귀를 표현하여 계산하는 것은 연속적인 두 값을 이용하여 나이브(naive)하게 반복적인 방법으로 대체될 수 있고, 이는 다시 연속적인 두 값을 단일 재귀의 매개변수로 넣음으로써 단일 재귀 방법으로 대체될 수 있다. 하지만 여기서의 단일 재귀는 일반 재귀 때 처럼 문제를 작은 사례로 분해하여 기본 사례로 가는 것이 아니라 처음부터 값을 추적해 나가는데, 이것을 **역 재귀(corecursion)**라고 부른다.

## Recursive programs

### **Factorial**

재귀의 전형적인 예로 팩토리얼 계산을 들 수 있다.

**Recursive procedures**

```c++
function factorial is:

input: integer n such that n >= 0

output: [n × (n-1) × (n-2) × … × 1]

    1. if n is 0, return 1
    2. otherwise, return [ n × factorial(n-1) ]

end factorial
```

---

위 팩토리얼을 계산하는 함수를 fact(n)로 보고 n을 매개 값으로 보면 n이 4일 때는 다음과 같이 계산이 된다.

```c++
fact(4) = 4 * fact(3)
        = 4 * (3 * fact(2))
        = 4 * (3 * (2 * fact(1)))
        = 4 * (3 * (2 * (1 * fact(0))))
        = 4 * (3 * (2 * (1 * 1)))
        = 4 * (3 * (2 * 1))
        = 4 * (3 * 2)
        = 4 * 6
        = 24
```

---

재귀 함수는 **루프 구조(looping construct)와 상수적 O(1)공간**을 요구하는 변수를 사용함으로써 재귀를 사용하지 않고 아래와 같이 **반복적인 방법(iterative)**으로 표현할 수 있다.

**Iterative procedures**

```c++
function factorial is:

input: integer n such that n >= 0

output: [n × (n-1) × (n-2) × … × 1]

    1. create new variable called running_total with a value of 1

    2. begin loop
          1. if n is 0, exit loop
          2. set running_total to (running_total × n)
          3. decrement n
          4. repeat loop

    3. return running_total

end factorial
```

### Greatest common divisor

두 정수의 최대 공약수(GCD, Greatest common divisor)를 계산하는 유클리디언 알고리즘도 다음과 같이 재귀를 이용하여 표현될 수 있다.

**Recursive procedures**

```c++
function gcd is:
input: integer x, integer y such that x > 0 and y >= 0

    1. if y is 0, return x
    2. otherwise, return [ gcd( y, (remainder of x/y) ) ]

end gcd
```

---

x = 111 이고 y = 259 일 때의 위 함수를 적용하여 다음과 같이 계산된다.

```c++
gcd(111, 259)   = gcd(259, 111 % 259)
                = gcd(259, 111)
                = gcd(111, 259 % 111)
                = gcd(111, 37)
                = gcd(37, 111 % 37)
                = gcd(37, 0)
                = 37
```

---

마찬가지로 반복적인 방법으로 표현될 수 있다.

**Iterative procedures**

```c++
function gcd is:

input: integer x, integer y such that x >= y and y >= 0

    1. create new variable called remainder

    2. begin loop
          1. if y is zero, exit loop
          2. set remainder to the remainder of x/y
          3. set x to y
          4. set y to remainder
          5. repeat loop

    3. return x

end gcd
```

### Binary search

이진 탐색은 사전에 미리 정렬이 완료된 배열이나 리스트에서 특정 값을 찾는 탐색 방법으로써 탐색의 매 단계마다 입력 배열의 반틈을 반복적으로 검색해 나간다. 입력 배열의 중간 지점을 선택해 찾고자 하는 데이터와 비교하고 찾았다면 그 값을 리턴한다. 아니라면, 중간 지점의 데이터가 찾고자 하는 데이터보다 작다면 중간 지점의 오른쪽 배열 반틈을, 중간 지점의 데이터가 찾고자 하는 데이터보다 크다면 중간 지점의 왼쪽 배열 반틈을 다시 검색해 나간다.

재귀의 속성이 이진 탐색에서 활용될 수 있는데, 매 탐색의 단계마다 입력 배열의 반틈을 반복적으로 탐색해 나가기 때문이다. 탐색의 매 단계마다 이전 배열로 부터 만들어진 새로운 배열에 대해 이진 탐색을 하는데 있어 재귀 함수를 적용할 수 있다. 이진 탐색은 탐색의 매 단계 마다 문제의 크기를 반틈씩 나누므로 로그적$O(logn)$ 성능을 보인다.

다음은 재귀를 적용한 이진 탐색 코드이다.

```c++
/*
  Call binary_search with proper initial conditions.

  INPUT:
    data is an array of integers SORTED in ASCENDING order,
    toFind is the integer to search for,
    count is the total number of elements in the array

  OUTPUT:
    result of binary_search

 */
 int search(int *data, int toFind, int count)
 {
    //  Start = 0 (beginning index)
    //  End = count - 1 (top index)
    return binary_search(data, toFind, 0, count-1);
 }

 /*
   Binary Search Algorithm.

   INPUT:
        data is a array of integers SORTED in ASCENDING order,
        toFind is the integer to search for,
        start is the minimum array index,
        end is the maximum array index
   OUTPUT:
        position of the integer toFind within array data,
        -1 if not found
 */
 int binary_search(int *data, int toFind, int start, int end)
 {
    //Get the midpoint.
    int mid = start + (end - start)/2;   //Integer division

    //Stop condition.
    if (start > end)
       return -1;
    else if (data[mid] == toFind)        //Found?
       return mid;
    else if (data[mid] > toFind)         //Data is greater than toFind, search lower half
       return binary_search(data, toFind, start, mid-1);
    else                                 //Data is less than toFind, search upper half
       return binary_search(data, toFind, mid+1, end);
 }
```

## References

- [Recursion](https://en.wikipedia.org/wiki/Recursion)
- [Recursion (Computer science)](<https://en.wikipedia.org/wiki/Recursion_(computer_science)>)
- [Functional programming](https://en.wikipedia.org/wiki/Functional_programming)
