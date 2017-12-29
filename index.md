---
layout: post
---
This is the homepage of my blogs.
<ul>
  {% for post in site.posts %}
    <li>
      <a href="/amitoj-blogs{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
