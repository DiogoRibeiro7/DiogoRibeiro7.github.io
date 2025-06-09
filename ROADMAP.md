# Roadmap de Evolução do Site

Este roadmap foi pensado em **sprints quinzenais** (2 semanas), para manter ritmo ágil e entregáveis claros.

---

## 📌 Visão Geral das Sprints

| Sprint | Duração       | Foco Principal                     |
|--------|---------------|------------------------------------|
| 1      | 12–25 Maio    | Fundamentos de Design              |
| 2      | 26 Maio–8 Jun | Layout e Navegação                 |
| 3      | 9–22 Jun      | Funcionalidades Avançadas          |
| 4      | 23 Jun–6 Jul  | Otimização e Acessibilidade        |
| 5      | 7–20 Jul      | QA, Monitorização e Lançamento     |

---

## Sprint 1 (12–25 Maio) – Fundamentos de Design

### Objetivo  
Criar identidade visual forte e consistente.

### Tarefas  
1. **Audit de Design atual**  
   - Revisar cores, tipografia e espaçamentos  
   - Gerar moodboard com referências de UI  

2. **Tipografia**  
   - Integrar Google Fonts (“Inter” + “Nunito Sans”)  
   - Ajustar variáveis SCSS:  
     ```scss
     $body-font-family: 'Inter', system-ui, sans-serif;
     $heading-font-family: 'Nunito Sans', system-ui, sans-serif;
     $base-font-size: 1rem;
     $h1-size: 2.75rem; // +10%
     ```

3. **Paleta de Cores**  
   - Definir 5 cores principais (primary, secondary, bg, surface, accent)  
   - Atualizar `_sass/minimal-mistakes/_variables.scss`  

4. **Protótipo de Homepage**  
   - Wireframe em Figma/Sketch  
   - Aprovação rápida antes de codificar  

### Entregáveis  
- Moodboard e esquema de cores  
- SCSS de variáveis pronto e testado localmente  
- Protótipo de homepage validado

---

## Sprint 2 (26 Maio–8 Junho) – Layout & Navegação

### Objetivo  
Reestruturar homepage e menu para melhor UX.

### Tarefas  
1. **“Splash” ou “Showcase” na Homepage**  
   - Front-matter `layout: home` + `home.splash`  
   - Imagem hero responsiva  

2. **Grid de Conteúdos / Features**  
   - Definir 4–6 blocos de destaque  
   - Implementar CSS Grid para responsividade  

3. **Menu Sticky & Mega-Menu**  
   - CSS SCSS para `position: sticky` + backdrop  
   - Estruturar `_data/navigation.yml` com categorias e subitens  

4. **Sidebar Dinâmica**  
   - Habilitar sidebar em `_config.yml`  
   - Incluir tags populares, posts relacionados, call-to-action de newsletter  

### Entregáveis  
- Homepage redesenhada e responsiva  
- Menu e sidebar funcionando em desktop e mobile  
- Checklist de responsividade validado

---

## Sprint 3 (9–22 Junho) – Funcionalidades Avançadas

### Objetivo  
Adicionar interatividade e usabilidade extra.

### Tarefas  
1. **Modo Claro / Escuro**  
   - Config `_config.yml`:  
     ```yaml
     color_scheme:
       default: light
       alternate: dark
     ```  
   - Botão-toggle e persistência com localStorage  

2. **Busca Full-text**  
   - Integrar Lunr.js (ou Algolia, se tiver conta)  
   - Campo de pesquisa no header e página de resultados  

3. **Galeria e Lightbox**  
   - Plugin Magnific Popup ou PhotoSwipe  
   - Estilos de hover e legenda overlay  

4. **Comentários via Utterances**  
   - Script Utterances (comentários GitHub)  
   - Ajustar fluxo de moderação

### Entregáveis  
- Dark mode funcional em todos os layouts  
- Busca indexando títulos e conteúdo  
- Galeria de imagens com lightbox  
- Seção de comentários ativa

---

## Sprint 4 (23 Junho–6 Julho) – Performance & Acessibilidade

### Objetivo  
Garantir carregamento rápido e conformidade WCAG.

### Tarefas  
1. **Otimização de Assets**  
   - Minificar CSS/JS (Rakefile)  
   - Converter imagens para WebP + lazy-loading  

2. **SEO Básico**  
   - Meta tags Open Graph e Twitter Cards  
   - Sitemap.xml e robots.txt  

3. **Acessibilidade (a11y)**  
   - Testes com axe-core  
   - Revisar landmarks, alt texts, navegação via teclado  

4. **Monitorização**  
   - Google Analytics / Plausible  
   - Configurar metas de conversão (newsletter, tempo em página)

### Entregáveis  
- Relatório de performance (Lighthouse)  
- Checklist WCAG 2.1 atendido  
- Painel de analytics inicial

---

## Sprint 5 (7–20 Julho) – QA, Lançamento & Feedback

### Objetivo  
Testar, lançar e planejar iterações futuras.

### Tarefas  
1. **Testes Finais**  
   - Cross-browser (Chrome, Firefox, Safari, Edge)  
   - Teste em dispositivos mobile reais  

2. **Deploy de Produção**  
   - `JEKYLL_ENV=production bundle exec jekyll build`  
   - Publicar no GitHub Pages  

3. **Coleta de Feedback**  
   - Criar formulário (Google Forms / Typeform)  
   - Monitorar métricas 1ª semana pós-lançamento  

4. **Planejamento da Próxima Iteração**  
   - Analisar feedback e dados de uso  
   - Priorizar backlog para novas features

### Entregáveis  
- Site ao vivo em produção  
- Relatório de bugs e feedback inicial  
- Roadmap de iteração 2.0

---

> **Dica extra:** faz deploy contínuo via GitHub Actions para cada push na branch `main`, assim manténs sempre o site atualizado sem dor de cabeça.  
