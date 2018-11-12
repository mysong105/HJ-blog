---
layout: "post"
title: "Numerical Quadrature - 2. Orthogonalization and Gram-Schmidt"
author: "HJ-harry"
mathjax: true
---

# Numerical Quadrature - 2. Orthogonalization and Gram-Schmidt

## Intro
------------------

## Orthogonality
---
Let's say two **vectors** are orthogonal. We can easily picture this *geometrically* since most of us learned at least some of the subjects about vectors with geometrical interpretation. **Orthogonality** is not a concept that is bounded only to vectors. In fact, two *polynomials* can be orthogonal to each other, since it can also be represented by a vector. Sinusoidal functions are also orthogonal to each other if they have different frequencies. All these orthogonality can be defined with one simple rule. The inner product equals 0.
$$\int_{-1}^{1} f(x)g(x) \ = 0$$
In other words, the definition of orthogonal functions can be defined as above.  

## Gram-Schmidt Orthogonalization
---
If we have two vectors or polynomials which are not orthogonal, how do we make these orthogonal? Here kicks in the idea of **Gram-Schmidt Orthogonalization**. To easily visualize it, we can start with two vectors $\overrightarrow{a}$ and $\overrightarrow{b}$ forming an acute angle

![vector projection](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfqaY8zLczkpk2_GbRKg48YhHn9bve5cjIPsR01D_20ViU5jHkBw)

Taking dot product $\vec{a}\cdot\vec{b}$ projects one vector to another, in this case $\vec{a_1}$. When you subtract $\vec{a_1}$ from $\vec{a}$, We get a new vector,$\vec{a_2}$ which is orthogonal to $\vec{b}$. Put this into an equation, we can say that
$$\vec{a_2} = \vec{a} - \frac{(a,b)}{(b,b)}\times{\vec{b}}$$
Parantheses represent dot product, and the reason we are dividing it by dot product itself is because this turns vector $$\vec{b}$$ into a unit vector. Now that we found two **orthogonal basis**, we can keep producing bases(plural of basis) that are all perpendicular to each other. This is how **Gram-Schmidt Orthogonalization** works, and can be easily applied to polynomials expressed by vectors.

Starting with $$\vec{v_1}, \vec{v_2}, \vec{v_3}, ... \vec{v_n}$$,
$$\vec{u_1} = \vec{v_1}$$
$$\vec{u_2} = \vec{v_2} - \frac{(u_1,v_2)}{(u_1,u_1)}\times{\vec{u_1}}$$
$$\vec{u_3} = \vec{v_2} - \frac{(u_2,v_3)}{(u_2,u_2)}\times{\vec{u_2}} - \frac{(u_1,v_3)}{(u_1,u_1)}\times{\vec{u_1}}$$
This trend continues until you get as many bases as you want!  

The reason I brought up this post is because Gram-Schmidt Orthogonalization ultimately leads to deriving **Legendre polynomials** and ultimately, **Gaussian Quadrature**. We can use Gram-Schmidt Orthogonalization for bringing up orthogonal matrices and QR factorization, which are huge subjects in Linear Algebra. However, since it is much to cover in one post, I will cover this in another post about Linear Algebra.  

In the next post I will cover **Legendre polynomials** and why it is useful.
