---
layout: "post"
title: "Numerical Quadrature - Intro"
author: "HJ-harry"
mathjax: true
---

# Numerical Quadrature - Intro

## Intro
------------------
 **numerical quadrature** is a method for calculating area under a function which cannot be analytically calculated, or too cumbersome to do so. In many real life cases this is the case, and knowing techniques of numerical quadrature can be useful in numerous cases.  
**Integration** is a *multiplicative operation* of infinitesimal increments of *independent variables* summed continuously over an interval.  
$$I = g(x) = \int_a^bf(x)dx$$
We call $f(x)$ **integrand**, with a and b as **limits of integration**. However, the statement above can only be solved with analytical methods. Here, we want to do it with numerical methods.  

## Interpolation Function
With numerical quadrature, we do not use the original function, since we are not going to use this analytically, or rather it is impossible to do so. Instead, we are going to use **Interpolation function**, in other words **Interpolant** which is a function that *approximates* the original function.  
Interpolant passes through **nodes** which are arbitrary points of evaluation. If $$a = 1, b = 4$$ and there are three equally-spaced intervals, the following two nodes will be 2 and 3. Here, we can also see if there are $$n$$ intervals, there will be $$n-1$$ nodes.

![mid-point interval](http://tutorial.math.lamar.edu/Classes/CalcII/ApproximatingDefIntegrals_Files/image001.png)

Let's check the following example. There are 6 intervals, and like we expected there are 5 nodes. The easiest and the most erroneous way to make interpolants within these nodes is to interpolate it with zeroth-degree functions. This simply means that we are going to assume that on every intervals, there exists a constant function. If we denote interpolant from $x_2$ to $x_3$ as $p_2(x)$,
$p_2(x) = f(x_2 + (x_3 - x_2)/2)$
which looks fancy, but just means that it is a constant function with mid-point value of the corresponding interval. This method is called **Composite midpoint method**. the word *composite* means we divided intervals into subparts and summed it all up. This term will appear a lot as we dive further into more complex methods of numerical quadrature. Rest of the name *midpoint method* is quite self-explanatory.  
Like I said above, this method is the *easiest* and *the most erroneous*. We do like the fact that it is simple, but we cannot get approximate values with this method. To resolve that, we are going to use other interpolation methods which we will see on the following posts.  

## Reference
*King et al., Numerical and Statistical Methods for Bioengineering*
