---
title: "Articles by Category"
layout: page
permalink: /categories/
seo_title: "Articles by Category"
seo_description: "Every article grouped by category."
---

{% assign categories = site.categories | sort %}
<ul class="archive-index">
{% for category in categories %}
  <li><a href="#{{ category[0] | slugify }}">{{ category[0] }}</a> ({{ category[1].size }})</li>
{% endfor %}
</ul>

{% for category in categories %}
<h2 id="{{ category[0] | slugify }}">{{ category[0] }}</h2>
<ul class="archive-list">
{% for post in category[1] %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <span class="archive-date">{{ post.date | date: '%Y-%m-%d' }}</span></li>
{% endfor %}
</ul>
{% endfor %}
