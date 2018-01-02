---
layout: default
---
# [](#header-2)Welcome to Amitoj's Blogs
Here are some of my projects and musings which will make an interesting read for somoene interested in Deep Learning. You can contact me on [mail](mailto:amitoj96@gmail.com) for discussions.

<ul>
  {% for post in site.posts %}
    <li>
      <a href="/amitoj-blogs{{ post.url }}" target="_blank">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
