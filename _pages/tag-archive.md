---
title: "Articles by Tag"
layout: page
permalink: /tags/
seo_title: "Articles by Tag"
seo_description: "Every article grouped by tag."
---

{% assign tags = site.tags | sort %}
<ul class="archive-index">
{% for tag in tags %}
  <li><a href="#{{ tag[0] | slugify }}">{{ tag[0] }}</a> ({{ tag[1].size }})</li>
{% endfor %}
</ul>

{% for tag in tags %}
<h2 id="{{ tag[0] | slugify }}">{{ tag[0] }}</h2>
<ul class="archive-list">
{% for post in tag[1] %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <span class="archive-date">{{ post.date | date: '%Y-%m-%d' }}</span></li>
{% endfor %}
</ul>
{% endfor %}
