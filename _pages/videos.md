---
layout: page
permalink: /videos/
title: videos
description:
nav: true
nav_order: 3
---

<style>
  .videos-card {
    scroll-margin-top: 90px;
  }

  .videos-card:target,
  .videos-card.is-targeted {
    border-color: rgba(13, 110, 253, 0.35);
    box-shadow: 0 0 0 0.3rem rgba(13, 110, 253, 0.12), 0 1rem 2rem rgba(13, 110, 253, 0.12);
    animation: video-target-highlight 1.8s ease-out;
  }

  @keyframes video-target-highlight {
    0% {
      background-color: rgba(13, 110, 253, 0.12);
      box-shadow: 0 0 0 0.45rem rgba(13, 110, 253, 0.18), 0 1.2rem 2.4rem rgba(13, 110, 253, 0.16);
    }

    100% {
      background-color: transparent;
      box-shadow: 0 0 0 0 rgba(13, 110, 253, 0), 0 0 0 rgba(13, 110, 253, 0);
    }
  }
</style>

<script>
  document.addEventListener('DOMContentLoaded', () => {
    const storedAnchor = sessionStorage.getItem('selectedVideoAnchor');
    const hashAnchor = window.location.hash ? window.location.hash.slice(1) : null;
    const targetAnchor = storedAnchor || hashAnchor;

    if (!targetAnchor) return;

    const target = document.getElementById(targetAnchor);
    if (!target) return;

    if (storedAnchor) {
      sessionStorage.removeItem('selectedVideoAnchor');
    }

    document.querySelectorAll('.videos-card.is-targeted').forEach((card) => {
      card.classList.remove('is-targeted');
    });

    window.requestAnimationFrame(() => {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      target.classList.remove('is-targeted');
      void target.offsetWidth;
      target.classList.add('is-targeted');
      window.history.replaceState(null, '', `#${targetAnchor}`);
    });
  });
</script>

<div class="row">
  {% for video in site.data.videos %}
    {% assign video_anchor = video.title | slugify %}
    <div class="col-sm-12 col-lg-6 mb-4">
      <div class="card h-100 hoverable videos-card" id="{{ video_anchor }}">
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

          <div class="card-text">{{ video.description | markdownify }}</div>

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
