---
title: "Posts by Year"
permalink: /year-archive/
layout: archive
author_profile: true
---

<h1>Posts by Year</h1>
{% assign posts_by_year = site.posts | group_by_exp:"post", "post.date | date: '%Y'" %}
{% for year in posts_by_year %}
  <h2>{{ year.name }}</h2>
  <ul>
    {% for post in year.items %}
      <li>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        <small>{{ post.date | date: "%B %d, %Y" }}</small>
      </li>
    {% endfor %}
  </ul>
{% endfor %}
