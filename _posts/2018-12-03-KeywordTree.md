---
layout: "post"
title: "KeywordTree"
author: "HJ-harry"
mathjax: true
---

# KeywordTree
keyword tree 자료구조를 python을 이용해서 구현해보겠습니다.  

![example_keyword_tree](file:///example_keyword_tree.png)

## AhoNode Class
```python
class AhoNode:
    def __init__(self):
        self.goto = {}
        self.out = []
        self.fail = None
```
**AhoNode**라는 class는 **KeywordTree**와 **Aho-Corasick Tree**에서 사용됩니다. AhoNode class는 **goto, out, fail** 이라는 **attribute**를 갖고 있습니다.  다른 여느 tree 구조와 다르게 오늘 구현하는 trie(keyword tree, aho-corasick tree)에는 **parent attribute가 존재하지 않습니다**. 사용되는 method에 parent node를 따라 tree의 level을 역행해서 올라갈 필요가 없기 때문입니다. 각 attribute에 대한 설명은 다음과 같습니다.

1. goto
  leaf node가 아닌 node는 한 개 이상의 child node를 갖고 있습니다. 따라서 goto attribute는 이어지는 children node를 모두 담고 있는 dictionary 자료형입니다. 예시의 tree 구조 root node의 goto는 dictionary의 key로 'G', 'A', 'C'가 있습니다.  
 만약 leaf node라면 이 goto dictionary가 비어있게 되며, 이 dictionary가 비어있는지를 확인해서 해당 node가 leaf node인지 아닌지 판별하는 데에 쓰입니다.  

2. out
  keyword tree의 leaf node는 우리가 찾고자 하는 **어떤 pattern의 총 서열을 갖고 있습니다**. 예시 trie의 leaf node에는 **GGT, GGG, ATG, CG**가 있으며, 알고리즘을 통해 text에 해당 pattern이 존재한다는 것을 확인했을 경우 trie의 leaf에 도달했으며 해당 pattern을 찾았다라고 하는 output을 나타내야 합니다. Leaf node의 out attribute는 이 sequence들을 갖고 있으며, 이후 **naive_keyword_search**와 **keyword_search**에서 print에 활용합니다.  

3. fail
  fail attribute는 **keyword tree에서 이용되지 않습니다**. 이는 aho-corasick algorithm에서 이용될 attribute로 다음 문서에서 다루도록 하겠습니다.

## Aho_Create_Forest


```python
def aho_create_forest(patterns):
    root = AhoNode()

    for sequence in patterns:
        node = root
        for base in sequence:
            node = node.goto.setdefault(base, AhoNode())
        node.out.append(sequence)
    return root
```
Keyword Tree를 구성하기 위해, 우선 **Ahonode class에 해당하는 root node를  initialize합니다**. 주어진 sequence들 하나하나마다 root부터 시작하며 **setdefault**를 이용해서 염기를 하나씩 붙이는데, setdefault함수는 만약 dictionary에 해당 key가 없다면 새로운 염기를 **node.goto dictionary에 추가하고**, 있다면 **존재하는 그 node로 넘어가게 됩니다**.  

만약 sequence의 끝까지 iteration했다면 **leaf node에 도달했다**는 뜻이므로 leaf node의 **output attribute에 해당 sequence를 assign**합니다.

이를 존재하는 모든 sequence에 따라 반복하면 keywordTree가 완성됩니다.

##
