---
title: "Test draft"
author_profile: true
toc: true
toc_sticky: true
tags:
  - mathjax
date: 2016-11-14 08:15:00 +0530
categories: courses
header:
  teaser: "/assets/images/proj.jpg"
excerpt: "Test Page - Experimental"
description: "Test Page - Experimental"
og_image: "/assets/images/proj.jpg"
use_math: true
mathjax: true
---

# H1 Heading

## H2 Heading

### H3 Heading

---

Here's some basic text.

And here's some _italics_

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:

- First item

* Second item

- Third item

Here's a numbered list:

1. First
2. Second
3. Third

Python code block:

```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:

```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$

{% include figure class="centered" url="/assets/images/proj.jpg" image_path="/assets/images/perceptron/percept.jpg" alt="A two year old walking" %}

[here's a link!](http://www.today.com/parents/unborn-babies-are-hearing-you-loud-clear-8C11005474)

**Feedback**: Did you enjoy reading or think it can be improved? Don't forget to leave your thoughts in the comments section below! If you liked this article, please share it with your friends, and read a few more!
{: .notice--success}

# How to render math equations on your Minimal Mistakes

### Katerina Bosko, PhD

[view solution](https://www.cross-validated.com/How-to-render-math-on-Minimal-Mistakes/)

You need [MathJax](https://www.mathjax.org) if you want to render mathematical equations on your Minimal Mistakes website.

I first learned about it from [the blog post by Peter Willis](http://www.pwills.com/posts/2017/12/20/website.html) (thanks, Peter!) However, I couldn't enable MathJax even after following his instructions. So here's what I did.

### 1.

Add this snippet at the end of the `scripts.html` which is in the `_includes` folder:

```html
{% if page.mathjax %}
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script
  id="MathJax-script"
  defer
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
></script>
{% endif %}
```

The [MathJax documentation](https://www.mathjax.org/#gettingstarted) says that with this snippet you won't need to change anything with each new version of MathJax.

### 2.

Enable MathJax in `_config.yml` by adding `mathjax: true` to the page defaults

```yml
# Defaults
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: true
      mathjax: true
```

<i class="far fa-sticky-note"></i> **Note:** If you are using Docker, don't forget to rebuild your container after making changes to `_config.yml`.
{: .notice--info}
{: .text-justify}

```
docker build -t minimal-mistakes .
```

## Playing with Maths

Basically that's it. Try rendering \\(y_0\\) using `\\(` and `\\)` delimeters for inline equations.

Use `\\[` and `\\]` to display equations like this one:

\\[ f(a) = \frac{1}{2\pi i} \oint_\gamma \frac{f(z)}{z-a} dz \\]

LAtex

```latex
\\[ f(a) = \frac{1}{2\pi i} \oint_\gamma \frac{f(z)}{z-a} dz \\]
```

TeX

```tex
\\[ f(a) = \frac{1}{2\pi i} \oint_\gamma \frac{f(z)}{z-a} dz \\]
```

$$mean = \frac{\displaystyle\sum_{i=1}^{n} x_{i}}{n}$$

TeX/LaTeX

```tex
$$mean = \frac{\displaystyle\sum_{i=1}^{n} x_{i}}{n}$$
```

k*{n+1} = n^2 + k_n^2 - k*{n-1}

TeX/LaTeX

```tex
k_{n+1} = n^2 + k_n^2 - k_{n-1}
```

<i class="far fa-sticky-note"></i> **Note:** If things don't work, check if the MathJax script is added to the page using Developer tools.
{: .notice--info}
{: .text-justify}

# Great docs

[mmistakes](https://mmistakes.github.io/minimal-mistakes/docs/helpers/)

[mmistakes github file](https://github.com/mmistakes/minimal-mistakes/blob/master/docs/_posts/2010-09-09-post-gallery.md)

[mmistakes layouts](https://mmistakes.github.io/minimal-mistakes/docs/layouts/#splash-page-layout)

Trying galleries and and feature rows.

## Notices

Call attention to a block of text.

| Notice Type | Class              |
| ----------- | ------------------ |
| Default     | `.notice`          |
| Primary     | `.notice--primary` |
| Info        | `.notice--info`    |
| Warning     | `.notice--warning` |
| Success     | `.notice--success` |
| Danger      | `.notice--danger`  |

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice}` class.
{: .notice}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--primary}` class.
{: .notice--primary}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--info}` class.
{: .notice--info}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--warning}` class.
{: .notice--warning}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--success}` class.
{: .notice--success}

**Watch out!** This paragraph of text has been [emphasized](#) with the `{: .notice--danger}` class.
{: .notice--danger}

{% capture notice-text %}
You can also add the `.notice` class to a `<div>` element.

- Bullet point 1
- Bullet point 2
  {% endcapture %}

<div class="notice--info">
  <h4 class="no_toc">Notice Headline:</h4>
  {{ notice-text | markdownify }}
</div>
