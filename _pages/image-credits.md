---
layout: page
title: "Image Credits"
permalink: /image-credits/
author_profile: false
seo_title: "Image Credits"
seo_description: "Sources, authors and licences of the photographs used as article headers on this site."
---

The header photographs on this site come from Wikimedia Commons under licences that allow reuse. Each is listed here with its author, licence and source page; the remaining headers are generated compositions produced by the site's own scripts and carry no third-party rights.

<table>
  <thead>
    <tr><th>Image</th><th>Title</th><th>Author</th><th>Licence</th><th>Source</th></tr>
  </thead>
  <tbody>
    {% for c in site.data.image_credits %}
    <tr>
      <td><img src="{{ '/assets/images/headers/' | append: c.file | relative_url }}" alt="" width="160" height="90" loading="lazy" style="width: 10rem; height: auto;" /></td>
      <td>{{ c.title }}</td>
      <td>{{ c.author }}</td>
      <td>{{ c.licence }}</td>
      <td><a href="{{ c.source }}" rel="noopener">Wikimedia Commons</a></td>
    </tr>
    {% endfor %}
  </tbody>
</table>
