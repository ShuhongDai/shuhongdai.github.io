---
layout: page
permalink: /videos/
title: videos
description: A lightweight gallery for recorded demos, talks, and visual experiments.
nav: true
nav_order: 3
---

<div class="row">
  {% for video in site.data.videos %}
    <div class="col-sm-12 col-lg-6 mb-4">
      <div class="card h-100 hoverable">
        <div class="card-body">
          <div class="mb-3">
            {% include video.liquid
              path=video.source
              class="img-fluid rounded z-depth-1"
              controls=true
              poster=video.poster
            %}
          </div>

          <h4 class="card-title">{{ video.title }}</h4>

          <p class="card-text">{{ video.description }}</p>

          <p class="post-meta mb-2">
            {% if video.category %}
              <span>{{ video.category }}</span>
            {% endif %}
            {% if video.category and video.date %}
              &nbsp; &middot; &nbsp;
            {% endif %}
            {% if video.date %}
              <span>{{ video.date | date: "%B %d, %Y" }}</span>
            {% endif %}
            {% if video.platform %}
              {% if video.category or video.date %}
                &nbsp; &middot; &nbsp;
              {% endif %}
              <span>{{ video.platform }}</span>
            {% endif %}
          </p>

          {% if video.external_url %}
            <div>
              <a href="{{ video.external_url }}" class="btn btn-sm z-depth-0 ps-0" role="button">Open link</a>
            </div>
          {% endif %}
        </div>
      </div>
    </div>
  {% endfor %}
</div>
