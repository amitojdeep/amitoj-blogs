---
layout: default
---
# [](#header-2)About
Home!!
<ul>
  {% for post in site.posts %}
    <li>
      <a href="/amitoj-blogs{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
