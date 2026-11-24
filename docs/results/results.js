//The results visualizer. Everything shown comes from manifest.json, which
//experimental_results/build_results_site.py writes from the folder layout
//experimental_results/<section>/<dataset>/**.png at every docs build.

const MANIFEST_URL = "manifest.json";
const DROP_FOLDER = "experimental_results";
const SECTION_TOGGLE_LABEL = "Figures";
const DATASET_TOGGLE_LABEL = "Data set";

//groups listed here come first, in this order; the rest follow alphabetically
const GROUP_ORDER = ["", "joint_information", "correlation_vs_map_size", "run_time_vs_map_size",
                     "frac_diff_vs_map_size"];
//keyed by the lower-case folder name; the values are HTML (the parameter symbols)
const GROUP_TITLES = {
  joint_information: "Joint Information",
  loga: "ln(<i>A</i><sub>s</sub> * 10<sup>10</sup>)",
  omch2: "Ω<sub>c</sub><i>h</i><sup>2</sup>",
  theta_mc_100: "100 * θ<sub>MC</sub>",
  correlation_vs_map_size: "Cross Correlation",
  run_time_vs_map_size: "Runtime",
  frac_diff_vs_map_size: "Fractional Difference",
};

//the widest a figure is drawn, in rem, unless its rule says otherwise. The PNGs come in very
//different pixel sizes for the same font size, so the widths below are what makes their text
//and axes look alike next to each other
const DEFAULT_FIGURE_WIDTH = 62;

//[pattern on the file name without extension, description, width in rem (optional)]; a figure
//is ordered by the first rule it matches, and the description is its alt text (the figures
//carry their own titles, so nothing is printed under them)
const FIGURE_RULES = [
  [/^z_scores$/, "Z-score, R-hat and burn-in table", 80],
  [/^triangle_plot$/, "Triangle plot", 44],
  [/_all_chain_marginals$/, "Per-map posteriors", 32],
  [/_all_chain_trace$/, "Per-map traces", 64],
  [/mean.distributions$/i, "Product distribution", 52],
  [/_integrated_autocorrelation_times$/, "Integrated autocorrelation times", 32],
  [/_auto_correlation$/, "Autocorrelation", 32],
  [/^cross_correlation_(.+)_all_N(\d+)$/, "Cross correlation, $1, N = $2"],
  [/^cached_run_time_/, "Cached run time"],
  [/^frac_diff_sim_vs_pred_.*$/, "Fractional difference between the simulated and the reconstructed fields"],
  [/^phi_noise_spectra_[\d.]+uk$/, "Phi noise spectra"],
  [/^phi_noise_spectra_over_qe_[\d.]+uk$/, "Phi noise spectra relative to the QE"],
  [/^sampler_rhat_[\d.]+uk$/, "Sampler convergence (R-hat)"],
];

//figures of one group drawn side by side in one row, left to right
const FIGURE_ROWS = [
  [/_auto_correlation$/, /_integrated_autocorrelation_times$/],
];

//sections whose groups (e.g. joint information and one per parameter) can be collapsed
const COLLAPSIBLE_SECTIONS = ["lcdm_sampling", "map_joint_comparison"];
//the heading of a collapsible section's figures that sit in no sub-folder
const UNGROUPED_TITLE = "Joint Information";

const togglesRoot = document.getElementById("toggles");
const sectionRoot = document.getElementById("sections");
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");

let manifest = null;
//the section shown and each section's own data set
const state = { section: null, datasets: {} };

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function baseName(file) {
  return file.split("/").pop().replace(/\.png$/i, "");
}

function figureRank(file) {
  const rank = FIGURE_RULES.findIndex(([pattern]) => pattern.test(baseName(file)));
  return rank === -1 ? FIGURE_RULES.length : rank;
}

function figureDescription(file) {
  const name = baseName(file);
  const rule = FIGURE_RULES.find(([pattern]) => pattern.test(name));
  //a description may refer to the pattern's groups as $1, $2
  return rule ? name.match(rule[0])[0].replace(rule[0], rule[1]) : name.replace(/_/g, " ");
}

function figureWidth(file) {
  const rule = FIGURE_RULES.find(([pattern]) => pattern.test(baseName(file)));
  return rule && rule[2] ? rule[2] : DEFAULT_FIGURE_WIDTH;
}

function groupRank(name) {
  const rank = GROUP_ORDER.indexOf(name);
  return rank === -1 ? GROUP_ORDER.length : rank;
}

function compareNames(a, b) {
  return a.localeCompare(b, undefined, { numeric: true });
}

function activeSection() {
  return manifest.sections.find(section => section.key === state.section);
}

function activeDataset() {
  const section = activeSection();
  return section.datasets.find(dataset => dataset.name === state.datasets[section.key]);
}

//---- the selection lives in the URL hash, so a view can be linked ----

function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const keys = manifest.sections.map(section => section.key);
  state.section = keys.includes(params.get("section")) ? params.get("section") : keys[0];
  for (const section of manifest.sections) {
    const names = section.datasets.map(dataset => dataset.name);
    const wanted = params.get(`dataset.${section.key}`);
    state.datasets[section.key] = names.includes(wanted) ? wanted : names[0];
  }
}

function writeHash() {
  const params = new URLSearchParams();
  params.set("section", state.section);
  for (const [key, name] of Object.entries(state.datasets)) {
    if (name) params.set(`dataset.${key}`, name);
  }
  history.replaceState(null, "", `#${params}`);
}

//---- the toggles, one column on the left ----

function optionButton(label, isActive, onClick) {
  const button = element("button", "option" + (isActive ? " active" : ""), label);
  button.type = "button";
  button.setAttribute("aria-pressed", isActive);
  button.addEventListener("click", onClick);
  return button;
}

function select(change) {
  Object.assign(state, change);
  writeHash();
  render();
  window.scrollTo(0, 0);
}

function toggleGroup(label, options) {
  const group = element("div", "toggle-group");
  const column = element("div", "options");
  column.append(...options);
  group.append(element("span", "toggle-label", label), column);
  return group;
}

function renderToggles() {
  const section = activeSection();
  const groups = [toggleGroup(SECTION_TOGGLE_LABEL, manifest.sections.map(candidate =>
    optionButton(candidate.title, candidate.key === section.key, () => select({ section: candidate.key }))))];
  if (section.datasets.length > 0) {
    groups.push(toggleGroup(DATASET_TOGGLE_LABEL, section.datasets.map(dataset =>
      optionButton(dataset.name, dataset.name === state.datasets[section.key],
                   () => select({ datasets: { ...state.datasets, [section.key]: dataset.name } })))));
  }
  togglesRoot.replaceChildren(...groups);
}

//---- figures ----

function figureBlock(dataset, file) {
  const description = figureDescription(file);
  const source = `${dataset.path}/${file.split("/").map(encodeURIComponent).join("/")}`;
  const figure = element("figure", "figure");
  figure.style.maxWidth = `min(100%, ${figureWidth(file)}rem)`;
  const button = element("button");
  button.type = "button";
  button.title = file;
  const image = element("img");
  image.src = source;
  image.alt = description;
  image.loading = "lazy";
  button.appendChild(image);
  button.addEventListener("click", () => {
    lightboxImage.src = source;
    lightboxImage.alt = description;
    lightbox.showModal();
  });
  figure.appendChild(button);
  return figure;
}

function groupBlocks(dataset, group, collapsible) {
  const files = [...group.files].sort((a, b) => figureRank(a) - figureRank(b) || compareNames(a, b));
  const blocks = [];
  const placed = new Set();
  for (const file of files) {
    if (placed.has(file)) continue;
    const patterns = FIGURE_ROWS.find(row => row.some(pattern => pattern.test(baseName(file))));
    const together = patterns
      ? patterns.map(pattern => files.find(other => pattern.test(baseName(other)))).filter(Boolean)
      : [file];
    together.forEach(member => placed.add(member));
    if (together.length === 1) {
      blocks.push(figureBlock(dataset, together[0]));
    } else {
      const row = element("div", "figure-row");
      row.append(...together.map(member => figureBlock(dataset, member)));
      blocks.push(row);
    }
  }
  const heading = element(collapsible ? "summary" : "h3");
  if (group.name.toLowerCase() in GROUP_TITLES) {
    heading.innerHTML = GROUP_TITLES[group.name.toLowerCase()];
  } else {
    heading.textContent = group.name || UNGROUPED_TITLE;
  }
  if (collapsible) {
    //closed until clicked
    const details = element("details", "group");
    details.append(heading, ...blocks);
    return [details];
  }
  if (group.name) blocks.unshift(heading);
  return blocks;
}

function note(...children) {
  const box = element("p", "note");
  box.append(...children);
  return box;
}

function sectionBody(section) {
  const dataset = activeDataset();
  if (!dataset) {
    return [note("No data sets yet. Drop a folder of figures into ",
                 element("code", null, `${DROP_FOLDER}/${section.key}/`), ".")];
  }
  const blocks = [];
  if (dataset.summary_html) {
    //the data set's SUMMARY.md, rendered to HTML by build_results_site.py
    const summary = element("div", "summary");
    summary.innerHTML = dataset.summary_html;
    blocks.push(summary);
  }
  const groups = [...dataset.groups].sort((a, b) =>
    groupRank(a.name) - groupRank(b.name) || compareNames(a.name, b.name));
  const collapsible = COLLAPSIBLE_SECTIONS.includes(section.key);
  return [...blocks, ...groups.flatMap(group => groupBlocks(dataset, group, collapsible))];
}

function render() {
  renderToggles();
  const section = activeSection();
  const block = element("section", "section");
  block.append(element("h2", null, section.title), ...sectionBody(section));
  sectionRoot.replaceChildren(block);
}

//a click on the image or on the backdrop closes the full-size view (Esc is built in)
lightbox.addEventListener("click", () => lightbox.close());

window.addEventListener("hashchange", () => {
  readHash();
  render();
});

fetch(MANIFEST_URL)
  .then(response => response.json())
  .then(loaded => {
    manifest = loaded;
    //2.5uk_... before 10uk_...
    manifest.sections.forEach(section => section.datasets.sort((a, b) => compareNames(a.name, b.name)));
    readHash();
    render();
  })
  .catch(() => {
    sectionRoot.replaceChildren(note(
      "manifest.json could not be loaded. Build the site with mkdocs build or mkdocs serve."));
  });
