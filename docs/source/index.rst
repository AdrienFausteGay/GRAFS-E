GRAFS-E 1.0 — Specification
============================

.. image:: _static/logo.jpg
   :alt: GRAFS-E Logo
   :width: 300px
   :align: center

.. admonition:: Status of this document
   :class: note

   This is the **normative** specification for GRAFS-E v1.0 (tag ``v1.0.0``).

**GRAFS-E** (Generalized Representation of Agri-Food Systems — Extended) is a flexible, open-source Python model.
Version **1.0** covers **nitrogen** at annual resolution; **carbon, energy and other flows** are under active development.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Overview
      :class-card: sd-shadow-sm sd-rounded-2
      :link-type: doc
      :link: overview

      Boundaries and Research question adressed by GRAFS-E

      +++

      .. button-ref:: overview
         :ref-type: doc
         :color: primary

         Toward overview

   .. grid-item-card:: User Guide
      :class-card: sd-shadow-sm sd-rounded-2
      :link-type: doc
      :link: user-guide

      How to use the python package

      +++

      .. button-ref:: user-guide
         :ref-type: doc
         :color: primary

         To the user guide

   .. grid-item-card:: GRAFS-E Engine
      :class-card: sd-shadow-sm sd-rounded-2
      :link-type: doc
      :link: references/api
      
      Full description of GRAFS-E functionning

      +++

      .. button-ref:: grafs-e-engine
         :ref-type: doc
         :color: primary

         See math behind the beast

   .. grid-item-card:: Carbon layer
      :class-card: sd-shadow-sm sd-rounded-2
      :link-type: doc
      :link: C_layer

      Presentation of the beta carbon layer of GRAFS-E

      +++

      .. button-ref:: C_layer
         :ref-type: doc
         :color: primary

         Carbon layer doc

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Specification

   overview
   user-guide
   grafs-e-engine
   02-terms
   input
   output
   C_layer
   references/api