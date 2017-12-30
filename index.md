---
layout: default
---
# [](#header-2)Welcome to Amitoj's Blogs
Here are some of my recent musings and experiments which might be worth a read for you.
<ul>
  {% for post in site.posts %}
    <li>
      <a href="/amitoj-blogs{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
