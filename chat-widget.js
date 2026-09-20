/**
 * AI Maths Teacher — Embeddable Chat Widget
 *
 * Usage (drop into any HTML page):
 *
 *   <script
 *     src="http://localhost:8000/static/chat-widget.js"
 *     data-api="http://localhost:8000/api/v1"
 *     data-class="10"
 *     data-position="bottom-right"
 *   ></script>
 *
 * OR call manually after load:
 *   MathTeacherWidget.init({ api: 'http://...', classLevel: 10 });
 *
 * Config options (data-* attributes or init() object):
 *   api          — base URL of the API (default: http://localhost:8000/api/v1)
 *   class        — default class level 1-12
 *   position     — "bottom-right" | "bottom-left" (default: bottom-right)
 *   title        — widget header title (default: "AI Maths Teacher")
 */

(function () {
  'use strict';

  const DEFAULTS = {
    api: 'http://localhost:8000/api/v1',
    classLevel: null,
    position: 'bottom-right',
    title: 'AI Maths Teacher',
  };

  const CSS = `
    #mt-widget-btn {
      position: fixed;
      bottom: 24px;
      width: 52px; height: 52px;
      border-radius: 50%;
      background: #1a73e8;
      color: #fff;
      border: none;
      font-size: 22px;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(0,0,0,0.25);
      z-index: 99998;
      display: flex; align-items: center; justify-content: center;
    }
    #mt-widget-btn.right { right: 24px; }
    #mt-widget-btn.left  { left: 24px; }

    #mt-widget-panel {
      position: fixed;
      bottom: 88px;
      width: 360px;
      height: 500px;
      background: #fff;
      border: 1px solid #d0d0d0;
      border-radius: 8px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      display: flex;
      flex-direction: column;
      z-index: 99999;
      font-family: system-ui, sans-serif;
      font-size: 13px;
      overflow: hidden;
    }
    #mt-widget-panel.right { right: 24px; }
    #mt-widget-panel.left  { left: 24px; }
    #mt-widget-panel.hidden { display: none; }

    #mt-header {
      background: #1a73e8;
      color: #fff;
      padding: 10px 14px;
      display: flex; align-items: center; justify-content: space-between;
      flex-shrink: 0;
    }
    #mt-header span { font-weight: 600; font-size: 13px; }
    #mt-header button {
      background: none; border: none; color: #fff;
      font-size: 18px; cursor: pointer; line-height: 1; padding: 0;
    }

    #mt-settings {
      padding: 8px 12px;
      border-bottom: 1px solid #eee;
      display: flex; gap: 6px; align-items: center;
      flex-shrink: 0; flex-wrap: wrap;
      background: #fafafa;
    }
    #mt-settings label { font-size: 11px; color: #555; }
    #mt-settings select { font-size: 11px; padding: 3px 5px; border: 1px solid #ccc; border-radius: 3px; }
    .mt-mode {
      font-size: 11px; padding: 3px 8px; border: 1px solid #ccc;
      border-radius: 3px; cursor: pointer; background: #fff;
    }
    .mt-mode.active { background: #1a73e8; color: #fff; border-color: #1a73e8; }

    #mt-messages {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
      display: flex; flex-direction: column; gap: 10px;
    }
    .mt-msg { max-width: 85%; line-height: 1.5; }
    .mt-msg.user { align-self: flex-end; }
    .mt-msg.ai   { align-self: flex-start; }
    .mt-bubble {
      padding: 8px 11px; border-radius: 5px;
      white-space: pre-wrap; word-break: break-word; font-size: 13px;
    }
    .mt-meta { font-size: 10px; color: #999; margin-top: 3px; }
    .mt-chips { margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap; }
    .mt-chip {
      padding: 3px 8px; background: #e2e8f0; border: 1px solid #cbd5e1;
      border-radius: 10px; font-size: 10px; font-weight: 500; color: #1e293b;
      cursor: pointer; transition: all 0.15s ease;
    }
    .mt-chip:hover { background: #cbd5e1; color: #0f172a; }
    .mt-mcq-btn {
      display: flex; align-items: center; width: 100%; text-align: left;
      padding: 6px 10px; margin: 3px 0; background: #fff; border: 1px solid #cbd5e1;
      border-radius: 5px; font-size: 12px; color: #0f172a; cursor: pointer;
      transition: all 0.15s ease;
    }
    .mt-mcq-btn:hover { background: #f3e8ff; border-color: #9333ea; color: #581c87; }
    .mt-mcq-badge {
      display: inline-flex; align-items: center; justify-content: center;
      width: 18px; height: 18px; background: #f3e8ff; border: 1px solid #c084fc;
      color: #7e22ce; border-radius: 50%; font-weight: 700; margin-right: 8px;
      font-size: 10px; flex-shrink: 0;
    }

    .mt-thinking { display: flex; gap: 4px; padding: 8px 11px; background: #f1f3f4; border-radius: 5px; width: fit-content; }
    .mt-thinking span { width: 5px; height: 5px; border-radius: 50%; background: #999; animation: mt-blink 1.2s infinite; }
    .mt-thinking span:nth-child(2) { animation-delay: 0.2s; }
    .mt-thinking span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes mt-blink { 0%,80%,100%{opacity:0.3} 40%{opacity:1} }

    #mt-input-row {
      display: flex; gap: 6px; padding: 10px;
      border-top: 1px solid #eee; flex-shrink: 0;
    }
    #mt-q {
      flex: 1; padding: 7px 9px;
      border: 1px solid #ccc; border-radius: 4px;
      font-family: inherit; font-size: 13px;
      resize: none; min-height: 34px; max-height: 80px;
      line-height: 1.4;
    }
    #mt-q:focus { outline: none; border-color: #1a73e8; }
    #mt-send {
      padding: 0 12px; background: #1a73e8; color: #fff;
      border: none; border-radius: 4px; font-size: 13px; cursor: pointer;
    }
    #mt-send:disabled { background: #ccc; cursor: not-allowed; }
    #mt-status {
      font-size: 10px; color: #888; padding: 0 12px 6px;
      display: flex; align-items: center; gap: 4px; flex-shrink: 0;
    }
    .mt-dot { width: 6px; height: 6px; border-radius: 50%; background: #ccc; }
    .mt-dot.ok { background: #34a853; }
    .mt-dot.warn { background: #fbbc04; }
    .mt-dot.fail { background: #ea4335; }
  `;

  function init(opts) {
    const cfg = Object.assign({}, DEFAULTS, opts);

    // Inject styles once
    if (!document.getElementById('mt-widget-css')) {
      const style = document.createElement('style');
      style.id = 'mt-widget-css';
      style.textContent = CSS;
      document.head.appendChild(style);
    }

    // Inject KaTeX CSS & JS for math rendering
    if (!document.getElementById('mt-katex-css')) {
      const link = document.createElement('link');
      link.id = 'mt-katex-css';
      link.rel = 'stylesheet';
      link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css';
      document.head.appendChild(link);
    }
    if (!window.katex && !document.getElementById('mt-katex-js')) {
      const script1 = document.createElement('script');
      script1.id = 'mt-katex-js';
      script1.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js';
      document.head.appendChild(script1);
      const script2 = document.createElement('script');
      script2.id = 'mt-katex-autorender-js';
      script2.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js';
      document.head.appendChild(script2);
    }

    const pos = cfg.position === 'bottom-left' ? 'left' : 'right';

    // Panel
    const panel = document.createElement('div');
    panel.id = 'mt-widget-panel';
    panel.className = pos + ' hidden';
    panel.innerHTML = `
      <div id="mt-header">
        <span>${esc(cfg.title)}</span>
        <button onclick="document.getElementById('mt-widget-panel').classList.toggle('hidden'); document.getElementById('mt-widget-btn').textContent = document.getElementById('mt-widget-panel').classList.contains('hidden') ? '🎓' : '✕'">✕</button>
      </div>
      <div id="mt-settings">
        <label>Class:
          <select id="mt-cls">
            <option value="">Any</option>
            ${[8,9,10,11,12].map(n => `<option value="${n}"${cfg.classLevel==n?' selected':''}>${n}</option>`).join('')}
          </select>
        </label>
        <button class="mt-mode active" data-mode="INTERACTIVE">Interactive</button>
        <button class="mt-mode" data-mode="GUIDED">Guided</button>
        <button class="mt-mode" data-mode="HINT">Hint</button>
        <button class="mt-mode" data-mode="FULL_SOLUTION">Full</button>
      </div>
      <div id="mt-messages"></div>
      <div id="mt-status"><div class="mt-dot" id="mt-dot"></div><span id="mt-status-text">Checking…</span></div>
      <div id="mt-input-row">
        <textarea id="mt-q" rows="1" placeholder="Ask a maths question…"></textarea>
        <button id="mt-send">Send</button>
      </div>
    `;
    document.body.appendChild(panel);

    // Toggle button
    const btn = document.createElement('button');
    btn.id = 'mt-widget-btn';
    btn.className = pos;
    btn.textContent = '🎓';
    btn.title = 'AI Maths Teacher';
    btn.onclick = () => {
      panel.classList.toggle('hidden');
      btn.textContent = panel.classList.contains('hidden') ? '🎓' : '✕';
    };
    document.body.appendChild(btn);

    // State
    let mode = 'INTERACTIVE';
    let busy = false;
    let thinkId = 0;
    let chatHistory = [];

    // Mode buttons
    panel.querySelectorAll('.mt-mode').forEach(b => {
      b.onclick = () => {
        panel.querySelectorAll('.mt-mode').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        mode = b.dataset.mode;
      };
    });

    // Textarea auto-resize + Enter key
    const qEl = panel.querySelector('#mt-q');
    qEl.addEventListener('input', () => {
      qEl.style.height = 'auto';
      qEl.style.height = Math.min(qEl.scrollHeight, 80) + 'px';
    });
    qEl.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSend(); }
    });
    panel.querySelector('#mt-send').addEventListener('click', doSend);

    // Health check
    async function checkHealth() {
      const dot = panel.querySelector('#mt-dot');
      const txt = panel.querySelector('#mt-status-text');
      try {
        const r = await fetch(cfg.api + '/health');
        const d = await r.json();
        dot.className = 'mt-dot ' + (d.status === 'ok' ? 'ok' : 'warn');
        txt.textContent = d.status === 'ok' ? 'Server OK' : `DB: ${d.database}`;
      } catch {
        dot.className = 'mt-dot fail';
        txt.textContent = 'Server unreachable';
      }
    }
    checkHealth();
    setInterval(checkHealth, 30000);

    function formatMathText(s) {
      if (!s) return '';
      let str = String(s);
      str = str.replace(/-?\\?(?:d|t)?frac\{([^}]+)\}\{([^}]+)\}/gi, '($1/$2)');
      str = str.replace(/-?\\?(?:d|t)?frac(\d)(\d)/gi, '($1/$2)');
      str = str.replace(/\\?(?:d|t)?frac/gi, '');

      const sups = {'^0':'⁰','^1':'¹','^2':'²','^3':'³','^4':'⁴','^5':'⁵','^6':'⁶','^7':'⁷','^8':'⁸','^9':'⁹','^n':'ⁿ'};
      for (const [k, v] of Object.entries(sups)) {
        str = str.replaceAll(k, v);
      }
      str = str.replace(/\(\s*°\s*\)/g, '°');
      str = str.replace(/\^\s*\(?\s*°\s*\)?/g, '°');
      str = str.replace(/\\?perpendicular\b|\\?perp[\s\-]*endicular\b|⊥[\s\-]*endicular\b|perp[\s\-]*endicular\b/gi, 'perpendicular');
    str = str.replace(/\\perp\b|\bperp\b(?!\s*endicular)/gi, ' ⊥ ');
    str = str.replace(/\\?triangle\s+([A-Z]{1,3}\b)/gi, '△$1');
    str = str.replace(/\\?\bangle\b\s+([A-Z]{1,3}\b)/gi, '∠$1');
    str = str.replace(/\\cong|\bcong\b(?!\s*r?uent|\s*r?uence)/gi, ' ≅ ');
    str = str.replace(/\\[lL]ongrightarrow|\\[rR]ightarrow|\\implies/g, ' ⇒ ');
      str = str.replace(/\\[lL]ongleftarrow|\\[lL]eftarrow/g, ' ⇐ ');
      str = str.replace(/\\cdot/gi, '·').replace(/\\times/gi, '×');
      str = str.replace(/\\sqrt\{([^}]+)\}/gi, '√($1)').replace(/\\sqrt/gi, '√');
      str = str.replace(/\\text\{([^}]+)\}/gi, '$1');
      str = str.replace(/\\begin\{[^}]+\}|\\end\{[^}]+\}/gi, '');
      str = str.replace(/\\[\[\]\(\)]/g, '');
      str = str.replace(/&=(?:\s*)/g, '= ').replace(/&(?:\s*)/g, '').replace(/\\\\[^\n]*\n?/g, '\n');
      str = str.replace(/\\[;,:\!]|\\quad|\\qquad/g, ' ');

      str = str.replace(/\$\$(.*?)\$\$/gs, '$1').replace(/\$([^$\n]+)\$/g, '$1').replace(/\$/g, '');
      str = str.replace(/\\([a-zA-Z]+)/g, '$1');
      str = str.replace(/ {2,}/g, ' ');
      return str.trim();
    }

    function buildHtmlTable(header, rows) {
      let html = '<div style="overflow-x:auto; margin:10px 0;"><table style="width:100%; border-collapse:collapse; background:#ffffff; border:1px solid #cbd5e1; border-radius:6px; font-size:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05);">';
      if (header.length > 0) {
        html += '<thead style="background:#f1f5f9; color:#0f172a; font-weight:600;"><tr>';
        for (const h of header) {
          html += `<th style="padding:8px 12px; border:1px solid #cbd5e1; text-align:left;">${esc(h)}</th>`;
        }
        html += '</tr></thead>';
      }
      html += '<tbody>';
      for (let rIdx = 0; rIdx < rows.length; rIdx++) {
        const bg = rIdx % 2 === 0 ? '#ffffff' : '#f8fafc';
        html += `<tr style="background:${bg};">`;
        for (const cell of rows[rIdx]) {
          html += `<td style="padding:7px 12px; border:1px solid #e2e8f0; color:#334155;">${esc(cell)}</td>`;
        }
        html += '</tr>';
      }
      html += '</tbody></table></div>';
      return html;
    }

    function renderFormattedText(s) {
      if (!s) return '';
      let str = formatMathText(s);

      // Extract raw <svg>...</svg> blocks (or ```xml/svg/html code blocks containing <svg>, including unclosed) first
      const svgBlocks = [];
      str = str.replace(/(?:```(?:xml|svg|html)?\s*)?(<svg[\s\S]*?(?:<\/svg>|$))(?:\s*```)?/gi, (match, svgContent) => {
        let cleanSvg = svgContent.trim();
        if (cleanSvg && !cleanSvg.toLowerCase().includes('</svg>')) {
          cleanSvg += '</svg>';
        }
        svgBlocks.push(cleanSvg);
        return `\n___SVG_BLOCK_${svgBlocks.length - 1}___\n`;
      });

      // Extract code blocks ```...``` next to preserve text formatting
      const codeBlocks = [];
      str = str.replace(/```(?:[a-zA-Z]*)\n?([\s\S]*?)```/g, (match, code) => {
        codeBlocks.push(code);
        return `\n___CODE_BLOCK_${codeBlocks.length - 1}___\n`;
      });

      const lines = str.split('\n');
      const outLines = [];
      let inTable = false;
      let tableHeader = [];
      let tableRows = [];

      for (let i = 0; i < lines.length; i++) {
        const rawLine = lines[i];
        const line = rawLine.trim();

        if (line.startsWith('___SVG_BLOCK_') && line.endsWith('___')) {
          if (inTable) {
            outLines.push(buildHtmlTable(tableHeader, tableRows));
            inTable = false;
            tableHeader = [];
            tableRows = [];
          }
          const idx = parseInt(line.replace('___SVG_BLOCK_', '').replace('___', ''));
          const svg = svgBlocks[idx] || '';
          outLines.push(`<div style="margin:14px 0; display:flex; justify-content:center; background:#ffffff; border:1px solid #cbd5e1; border-radius:8px; padding:12px; box-shadow:0 1px 4px rgba(0,0,0,0.06); overflow-x:auto;">${svg}</div>`);
          continue;
        }

        if (line.startsWith('___CODE_BLOCK_') && line.endsWith('___')) {
          if (inTable) {
            outLines.push(buildHtmlTable(tableHeader, tableRows));
            inTable = false;
            tableHeader = [];
            tableRows = [];
          }
          const idx = parseInt(line.replace('___CODE_BLOCK_', '').replace('___', ''));
          const code = codeBlocks[idx] || '';
          outLines.push(`<pre style="background:#f1f5f9; color:#0f172a; border:1px solid #cbd5e1; border-radius:6px; padding:10px 14px; font-family:Consolas, Monaco, monospace; font-size:12px; line-height:1.45; overflow-x:auto; margin:10px 0; white-space:pre;">${esc(code)}</pre>`);
          continue;
        }

        if (line.startsWith('|') && line.endsWith('|')) {
          if (/^\|[\s\-:]+(\|[\s\-:]+)+\|$/.test(line)) {
            continue;
          }
          const cells = line.slice(1, -1).split('|').map(c => c.trim());
          if (!inTable) {
            inTable = true;
            tableHeader = cells;
            tableRows = [];
          } else {
            tableRows.push(cells);
          }
        } else {
          if (inTable) {
            outLines.push(buildHtmlTable(tableHeader, tableRows));
            inTable = false;
            tableHeader = [];
            tableRows = [];
          }
          outLines.push(line);
        }
      }
      if (inTable) {
        outLines.push(buildHtmlTable(tableHeader, tableRows));
      }

      const formattedBlocks = outLines.map(line => {
        if (line.startsWith('<div style="overflow-x:auto') || line.startsWith('<div style="margin:14px 0') || line.startsWith('<pre style=')) return line;

        let l = esc(line);
        if (/^###\s+((?:🌟\s*)?Quick\s*Quiz[\d\w–—\-:\s]*)/i.test(line)) {
          const title = line.replace(/^###\s+/, '');
          return `<div style="background:#f3e8ff; border-left:3px solid #9333ea; color:#581c87; padding:8px 12px; border-radius:5px; font-weight:600; margin:12px 0 6px; font-size:12px; display:flex; align-items:center; gap:6px;"><span>🧩</span> ${esc(title)}</div>`;
        }
        if (/^###\s+((?:Question:\s*)?Quick\s*Check[\d\w–—\-:\s]*)/i.test(line)) {
          const title = line.replace(/^###\s+/, '');
          return `<div style="background:#f0fdf4; border-left:3px solid #16a34a; color:#166534; padding:8px 12px; border-radius:5px; font-weight:600; margin:12px 0 6px; font-size:12px; display:flex; align-items:center; gap:6px;"><span>🎯</span> ${esc(title)}</div>`;
        }
        if (/^###\s+(Question\s*[\d\w–—\-:\s]*)/i.test(line)) {
          const title = line.replace(/^###\s+/, '');
          return `<div style="background:#eff6ff; border-left:4px solid #1a73e8; color:#1e40af; padding:8px 12px; border-radius:4px; font-weight:600; margin:14px 0 6px; font-size:13px; display:flex; align-items:center; gap:6px;"><span>📝</span> ${esc(title)}</div>`;
        }
        if (line.startsWith('### ')) {
          return `<h3 style="color:#1a73e8; font-size:14px; margin:12px 0 6px; font-weight:600;">${esc(line.slice(4))}</h3>`;
        }
        if (line.startsWith('## ')) {
          return `<h2 style="color:#1e293b; font-size:15px; margin:14px 0 8px; font-weight:700; border-bottom:1px solid #e2e8f0; padding-bottom:4px;">${esc(line.slice(3))}</h2>`;
        }
        if (line.startsWith('# ')) {
          return `<h1 style="color:#0f172a; font-size:16px; margin:16px 0 10px; font-weight:700;">${esc(line.slice(2))}</h1>`;
        }
        if (line === '---' || line === '***') {
          return '<hr style="border:0; border-top:1px solid #cbd5e1; margin:14px 0;">';
        }

        const mcqMatch = /^([A-D])[\.\)]\s+(.*)/i.exec(line);
        if (mcqMatch) {
          const optLetter = mcqMatch[1].toUpperCase();
          const optText = mcqMatch[2].trim();
          const sendPayload = `Option ${optLetter}: ${optText}`;
          const safePayload = esc(sendPayload).replace(/'/g, "\\'");
          return `<button class="mt-mcq-btn" onclick="document.querySelector('#mt-q').value='${safePayload}'; document.querySelector('#mt-send').click();"><span class="mt-mcq-badge">${esc(optLetter)}</span><span>${esc(optText)}</span></button>`;
        }

        l = l.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        l = l.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        return l;
      });

      return formattedBlocks.join('<br>');
    }

    function sendQuick(text) {
      qEl.value = text;
      doSend();
    }

    function addMsg(role, text, meta) {
      const msgs = panel.querySelector('#mt-messages');
      const w = document.createElement('div');
      w.className = 'mt-msg ' + role;

      let displayText = role === 'ai' ? renderFormattedText(text) : esc(text);

      const isSetupPrompt = text.includes('Mandatory Exam Preparation Settings') || text.includes('specify your exam preferences');

      const isExamPaper = role === 'ai' && !isSetupPrompt && (
        text.includes('Question Paper') ||
        text.includes('Exam Paper Ready!') ||
        (text.includes('Total Marks:') && (text.includes('[1 Mark]') || text.includes('[2 Marks]') || text.includes('[3 Marks]') || text.includes('[4 Marks]') || text.includes('[5 Marks]') || text.includes('[6 Marks]') || text.includes('[7 Marks]') || text.includes('CBSE Class')))
      );


      if (isExamPaper) {
        displayText += `
          <div class="mt-chips" style="margin-top:8px;">
            <button class="mt-chip" style="background:#1a73e8; color:#fff; font-weight:600; border:none;" onclick="
              const bubble = this.closest('.mt-bubble');
              if(!bubble) return;
              const clone = bubble.cloneNode(true);
              clone.querySelectorAll('.mt-chips, button').forEach(el=>el.remove());
              const win = window.open('','_blank');
              if(!win) return alert('Allow popups to download PDF');
              win.document.write('<html><head><title>Question Paper</title><style>body{font-family:sans-serif;padding:20px;}h1{color:#1a73e8;}svg{max-width:100%;}</style></head><body><h1>🎓 Question Paper</h1><div>'+clone.innerHTML+'</div><script>window.onload=function(){setTimeout(function(){window.print();},300);};<\/script></body></html>');
              win.document.close();
            ">📥 Download PDF</button>
          </div>
        `;
      }

      if (role === 'ai' && mode === 'INTERACTIVE' && !isExamPaper) {
        displayText += `
          <div class="mt-chips">
            <button class="mt-chip" onclick="document.querySelector('#mt-q').value='Give me a quick quiz on this concept!'; document.querySelector('#mt-send').click();">🧩 Quick Quiz</button>
            <button class="mt-chip" onclick="document.querySelector('#mt-q').value='Give me a hint for this question'; document.querySelector('#mt-send').click();">💡 Hint</button>
            <button class="mt-chip" onclick="document.querySelector('#mt-q').value='Show me the next step'; document.querySelector('#mt-send').click();">👉 Next Step</button>
            <button class="mt-chip" onclick="document.querySelector('#mt-q').value='Show the full worked solution'; document.querySelector('#mt-send').click();">📝 Full Solution</button>
          </div>
        `;
      }

      w.innerHTML = `<div class="mt-bubble">${displayText}</div>` +
        (meta ? `<div class="mt-meta">${esc(meta)}</div>` : '');
      msgs.appendChild(w);
      msgs.scrollTop = msgs.scrollHeight;

      if (window.renderMathInElement) {
        window.renderMathInElement(w, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true }
          ],
          throwOnError: false
        });
      }
    }

    function addThinking() {
      const id = 'mt-th' + (++thinkId);
      const msgs = panel.querySelector('#mt-messages');
      const w = document.createElement('div');
      w.className = 'mt-msg ai';
      w.id = id;
      w.innerHTML = `<div class="mt-thinking"><span></span><span></span><span></span></div>`;
      msgs.appendChild(w);
      msgs.scrollTop = msgs.scrollHeight;
      return id;
    }

    function setBusy(on) {
      busy = on;
      panel.querySelector('#mt-send').disabled = on;
      qEl.disabled = on;
    }

    async function doSend() {
      if (busy) return;
      const q = qEl.value.trim();
      if (!q) return;
      addMsg('user', q);
      qEl.value = '';
      qEl.style.height = 'auto';

      const body = { question: q, mode, chat_history: chatHistory.slice(-6) };
      const cls = panel.querySelector('#mt-cls').value;
      if (cls) body.class_level = parseInt(cls);

      const tid = addThinking();
      setBusy(true);

      try {
        const r = await fetch(cfg.api + '/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        panel.querySelector('#' + tid)?.remove();
        if (!r.ok) {
          const e = await r.json().catch(() => ({ detail: r.statusText }));
          addMsg('ai', '⚠ Error: ' + (e.detail || r.statusText));
          return;
        }
        const d = await r.json();
        chatHistory.push({ role: 'user', content: q });
        chatHistory.push({ role: 'assistant', content: d.answer });

        const meta = [
          d.intent?.replace(/_/g, ' '),
          d.grounding?.passed
            ? `✓ grounded ${(d.grounding.score * 100).toFixed(0)}%`
            : `⚠ low grounding`,
          d.math_verification?.status !== 'not_required'
            ? (d.math_verification?.passed ? '✓ math ok' : '✗ math fail')
            : null
        ].filter(Boolean).join(' · ');
        addMsg('ai', d.answer, meta, d.step_logs);
      } catch (err) {
        panel.querySelector('#' + tid)?.remove();
        addMsg('ai', '⚠ ' + (err.message === 'Failed to fetch'
          ? 'Cannot reach server.'
          : err.message));
      } finally {
        setBusy(false);
      }
    }
  }

  function esc(s) {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  // Auto-init from script tag data-* attributes
  function autoInit() {
    const scriptEl = document.currentScript ||
      document.querySelector('script[src*="chat-widget"]');
    if (!scriptEl) { window.MathTeacherWidget = { init }; return; }
    const cfg = {};
    if (scriptEl.dataset.api)      cfg.api = scriptEl.dataset.api;
    if (scriptEl.dataset.class)    cfg.classLevel = parseInt(scriptEl.dataset.class);
    if (scriptEl.dataset.position) cfg.position = scriptEl.dataset.position;
    if (scriptEl.dataset.title)    cfg.title = scriptEl.dataset.title;
    init(cfg);
    window.MathTeacherWidget = { init };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }
})();
