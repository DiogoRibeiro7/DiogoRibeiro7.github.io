---
title: "Articles by Category"
layout: page
permalink: /categories/
seo_title: "Articles by Category"
seo_description: "Every article grouped by category."
---

Categories group articles by broad subject. Use the [tag index]({{ '/tags/' | relative_url }}) to find languages, methods, and more specific topics.

{% assign categories = site.categories | sort %}
<ul class="archive-index">
{% for category in categories %}
  <li><a href="#{{ category[0] | slugify }}">{{ category[0] }}</a> ({{ category[1].size }})</li>
{% endfor %}
</ul>

<h2>Related topics</h2>
<p>These topics are now collected under tags, including articles previously listed only in other categories.</p>
<ul class="archive-index">
  <li id="r"><a href="{{ '/tags/' | relative_url }}#r">R</a></li>
  <li id="science-policy"><a href="{{ '/tags/' | relative_url }}#science-policy">Science Policy</a></li>
  <li id="software-engineering"><a href="{{ '/tags/' | relative_url }}#software-engineering">Software Engineering</a></li>
  <li id="statistical-computing"><a href="{{ '/tags/' | relative_url }}#statistical-computing">Statistical Computing</a></li>
</ul>

{% for category in categories %}
<h2 id="{{ category[0] | slugify }}">{{ category[0] }}</h2>
<ul class="archive-list">
{% for post in category[1] %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <span class="archive-date">{{ post.date | date: '%Y-%m-%d' }}</span></li>
{% endfor %}
</ul>
{% endfor %}
