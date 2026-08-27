# Third-party data and resources

This repository contains or makes use of data and resources originating from
third-party projects. These materials remain subject to the terms, licenses,
and attribution requirements of their original sources.

---

## PET receptor and transporter maps

The receptor and transporter maps stored under:

```text
codes/HCP/data/Receptor_maps/
```

are derived from the PET receptor-mapping resources distributed with the
`netneurolab/hansen_receptors` project associated with:

Hansen JY, Shafiei G, Markello RD, et al.  
**Mapping neurotransmitter systems to the structural and functional organization
of the human neocortex.**  
*Nature Neuroscience* (2022).

The Hansen receptor repository is distributed under the:

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0) license.**

Users of these receptor maps should cite:

1. Hansen et al. (2022); and
2. the original PET study or studies corresponding to each receptor or
   transporter map.

The original PET-study references are provided by the Hansen receptor-mapping
project and its accompanying supplementary material.

File names generally identify the receptor or transporter, PET tracer,
sample size, and source study, for example:

```text
5HT2a_cimbi_hc29_beliveau.csv
D2_fallypride_hc49_jaworska.csv
GABAa-bz_flumazenil_hc16_norgaard.csv
```

These third-party PET-derived data remain subject to the licensing and
attribution requirements of their original sources.

Source project:

```text
https://github.com/netneurolab/hansen_receptors
```

License:

```text
CC BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/
```

---

## Human Connectome Project data

The analyses in this repository use data from the WU-Minn Human Connectome
Project (HCP) Young Adult dataset.

The original HCP resting-state and task-fMRI data are **not redistributed**
in this repository.

Users wishing to reconstruct the analyses from source data should obtain the
required HCP data directly from the Human Connectome Project and accept the
applicable HCP Data Use Terms.

The HCP Open Access Data Use Terms permit redistribution of Open Access and
derived data only when they are redistributed under the same HCP Data Use
Terms. To avoid redistributing subject-level HCP resources outside that access
framework, this repository does not include the HCP task event/timing files.

For the task analyses, users should obtain the required HCP task timing files
directly from HCP and place them under:

```text
codes/HCP/data/Task/HCP_TASKS_EVs/
```

The analysis code expects this directory structure when the task-analysis
pipeline is rerun from source data.

The paper cohort identifier list is stored separately in:

```text
codes/HCP/data/Task/language_subjects_paper100.txt
```

The original HCP imaging data remain external to this repository.

HCP data-use information:

```text
https://www.humanconnectome.org/study/hcp-young-adult/document/extensively-processed-fmri-data-documentation
```

Users should consult the current Human Connectome Project website and data-use
terms before downloading or using HCP data.

---

## HCP acknowledgement

Publications or presentations using WU-Minn HCP data should follow the
acknowledgement requirements specified by the Human Connectome Project.

The HCP Open Access Data Use Terms provide the following acknowledgement:

> Data were provided [in part] by the Human Connectome Project, WU-Minn
> Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil;
> 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the
> NIH Blueprint for Neuroscience Research; and by the McDonnell Center for
> Systems Neuroscience at Washington University.

Users should verify the current HCP acknowledgement requirements when
publishing work based on HCP data.

---

## Atlases and cortical surface resources

The repository uses Schaefer cortical parcellation and Tian subcortical
parcellation labels and associated atlas resources for analysis and
visualization.

Files used by the figure code include resources under:

```text
codes/figures/_atlases/
```

and:

```text
codes/HCP/data/
```

These atlas resources remain subject to the citation and licensing requirements
of their respective original publications and distributions.

The surface-plotting code may additionally obtain fsLR surface resources through
the `neuromaps` Python package on first use. These downloaded resources are not
stored in this repository.

---

## Software dependencies

Third-party Python packages used by this project are listed in:

```text
requirements.txt
```

Each package remains subject to its own software license.

---

## Scope of this notice

This file documents third-party data and resources used by the project.

A software license applied to the code in this repository does **not**
supersede or replace the licenses, data-use terms, or attribution requirements
that apply to third-party datasets or resources.

Users are responsible for ensuring that their use and redistribution of
third-party materials complies with the applicable terms of the original
sources.
