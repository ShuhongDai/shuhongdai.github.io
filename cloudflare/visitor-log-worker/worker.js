export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders() });
    }

    if (url.pathname === '/collect' && request.method === 'POST') {
      return handleCollect(request, env);
    }

    if (url.pathname === '/admin' && request.method === 'GET') {
      return handleAdmin(request, env, url);
    }

    if (url.pathname === '/health') {
      return json({ ok: true }, 200);
    }

    return json({ error: 'Not found' }, 404);
  },
};

async function handleCollect(request, env) {
  if (!env.DB) {
    return json({ error: 'Missing DB binding' }, 500);
  }

  let body = {};
  try {
    body = await request.json();
  } catch {
    body = {};
  }

  const city = request.cf?.city || 'Unknown';
  const country = request.cf?.country || 'Unknown';
  const path = typeof body.path === 'string' ? body.path.slice(0, 300) : '/';
  const site = typeof body.site === 'string' ? body.site.slice(0, 120) : 'default';
  const ts = new Date().toISOString();

  await env.DB.prepare(
    `INSERT INTO visits (ts, city, country, path, site)
     VALUES (?1, ?2, ?3, ?4, ?5)`
  )
    .bind(ts, city, country, path, site)
    .run();

  return new Response(null, {
    status: 204,
    headers: corsHeaders(),
  });
}

async function handleAdmin(request, env, url) {
  if (!env.ADMIN_TOKEN || url.searchParams.get('token') !== env.ADMIN_TOKEN) {
    return json({ error: 'Unauthorized' }, 401);
  }

  const limit = Math.min(Number(url.searchParams.get('limit') || '100'), 500);
  const format = url.searchParams.get('format') || 'html';

  const { results } = await env.DB.prepare(
    `SELECT ts, city, country, path, site
     FROM visits
     ORDER BY id DESC
     LIMIT ?1`
  )
    .bind(limit)
    .all();

  if (format === 'json') {
    return json({ visits: results }, 200);
  }

  const rows = results
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.ts || '')}</td>
          <td>${escapeHtml(row.city || '')}</td>
          <td>${escapeHtml(row.country || '')}</td>
          <td>${escapeHtml(row.path || '')}</td>
          <td>${escapeHtml(row.site || '')}</td>
        </tr>`
    )
    .join('');

  return new Response(
    `<!doctype html>
      <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Private Visit Log</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; color: #111; }
            h1 { margin-bottom: 1rem; }
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 0.65rem 0.75rem; border-bottom: 1px solid #ddd; text-align: left; font-size: 0.95rem; }
            th { background: #f7f7f7; }
            code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 4px; }
          </style>
        </head>
        <body>
          <h1>Private Visit Log</h1>
          <p>Recent ${results.length} visits.</p>
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>City</th>
                <th>Country</th>
                <th>Path</th>
                <th>Site</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </body>
      </html>`,
    {
      headers: {
        'content-type': 'text/html; charset=utf-8',
        'cache-control': 'no-store',
      },
    }
  );
}

function json(data, status) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      ...corsHeaders(),
    },
  });
}

function corsHeaders() {
  return {
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'POST, GET, OPTIONS',
    'access-control-allow-headers': 'Content-Type',
  };
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
