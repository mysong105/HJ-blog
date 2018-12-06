---
layout: "post"
title: "Aho-Corasick algorithm"
author: "HJ-harry"
mathjax: true
---

## Aho-Corasick-Tree
이번엔 **Keyword-Tree**와 비슷한 방식으로 **Aho-Corasick-Tree**를 구현해보겠습니다.

![Imgur](https://i.imgur.com/4Rkr7o8.png)

## AhoNode Class

**Keyword-Tree**와 동일한 class structure를 사용하기 때문에 많은 추가적인 설명은 필요하지 않습니다. 다만 이번에는 **fail** attribute를 활용합니다. 위 그림과 같이 failure link는 root를 제외한 모든 node에 존재하기 때문에 각 node.fail에 failure link로 이어지는 node를 assign해줍니다. 따라서 text를 따라 search를 진행할 때, 끊김 없이 tree를 계속 따라 이동할 수 있습니다.

## Aho_create_tree
```python
def aho_create_tree(patterns):
    root = make_keyword_tree(patterns)
    queue = []

    # For nodes that are children of root node
    # Link their failure nodes to root
    for _,node in root.goto.items():
        queue.append(node)
        node.fail = root

    while len(queue) > 0:
        pnode = queue.pop(0)

        for key, qnode in pnode.goto.items():
            queue.append(qnode)
            fnode = pnode.fail
            while fnode != None and not key in fnode.goto:
                fnode = qnode.fail
            qnode.fail = fnode.goto[key] if fnode else root
            qnode.out += qnode.fail.out

    return root
```
먼저 failure link를 제외한 goto link와 outputlink는 기존 keyword tree 구조에서 존재하기 때문에 먼저 make_keyword_tree 함수를 이용해서 failuer link를 제외한 tree를 구성합니다.  

구성한 이후, root node부터 다시 들어가면서 Breadth first search를 하기 위해 level by level로 들어가며 있는 node들을 queue에 넣습니다. 예컨대 예시 trie에서 첫 번째 level에 있는 [A,G,C]가 queue에 들어가고 A가 pop하며 이 것의 goto node인 [C,T]가 queue에 들어가는 식입니다.  

엄밀히 말하면 keyword tree를 구성하고 한 번 더 root부터 따라 들어가는 것이기에 제가 구현한 코드는 O(N)이 아니라 O(2*N)이라고 볼 수 있습니다. 하지만 aho-corasick algorithm을 보여주는 데에 초점을 맞췄고 N의 수가 practical한 상황에서 전체 염기 서열의 길이인 m에 비하면 trivial하다는 관점에서 큰 문제는 없다고 생각할 수 있습니다.

Failure link가 연결되는 법칙에 따라 p node의 children에 q node와 같은 base가 있다면 failure link가 그리로 연결이 되며, 없어서 failure link를 따라 올라가다 root node에 도달한다면 failure link를 root node로 연결을 합니다. output link가 존재하는 node에 한해서 해당 node에서 failure link가 다른 root가 아닌 node에 연결된 경우 **p node의 output에 p node의 failure link에 존재하는 output을 추가해줍니다**.  

## Aho_search

```python
def aho_search(root,DNAsequence):

    node = root

    for base in DNAsequence:
        if base in node.goto:
            node = node.goto[base]
            # If there exists output link
            if node.out != []:
                # Since there might be more than one output sequence
                for patterns in node.out:
                    print("Found a pattern! - ")
                    print(patterns)

        else:
            # Move via failure link
            while base not in node.goto:

                node = node.fail
                if base in node.goto:
                    node = node.goto[base]
                    if node.out != []:
                    # Since there might be more than one output sequence
                        for patterns in node.out:
                            print("Found a pattern! - ")
                            print(patterns)
                    break

                if node is root:
                    break

    return
```

Aho_create_tree 함수를 통해 만들어진 trie 구조를 통해 주어진 DNA sequence를 찾는 알고리즘입니다. 우선 input으로 root node와 연결된 trie, 그리고 찾고자하는 DNA sequence들이 들어갑니다. Base가 동일하다면 goto를 통해 다음 level로 넘어가고, 다르다면 failure link를 따라 다른 node로 이어져 그 node의 goto를 조사합니다. 만약 root node까지 왔는데도 goto에 해당하는 염기서열이 없다면, 다음 iteration으로 넘어가게 됩니다.
