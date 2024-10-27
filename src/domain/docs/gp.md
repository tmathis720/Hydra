## instructions

Below, find reference articles which describe the code we are trying to develop in Rust for the HYDRA program. Using the code provided below the reference articles and in subsequent prompts, prepare a detailed outline documenting the `Domain` module as described by the code provided in this prompt. I will inform you when I have provided all of the components of the `Domain` module.

In subsequent prompts, I will request detailed, complete versions of each section identified. You will then provide a complete response containing the entirety of the section requested, and only that section. Maintain an accurate memory of the code provided, as well as the articles to be a primary source throughout the preparation of the report. Do not halucinate or speculate about details of what you have been provided.

## reference articles

> Mesh Algorithms for PDE with Sieve I: Mesh

Distribution

Matthew G. Knepley

Computation Institute

University of Chicago

Chicago, IL 60637

Dmitry A. Karpeev

Mathematics and Computer Science Division

Argonne National Laboratory

Argonne, IL 60439

October 23, 2018

arXiv:0908.4427v1 \[cs.CE\] 30 Aug 2009

# Abstract {#abstract .unnumbered}

> We have developed a new programming framework, called Sieve, to
> support parallel numerical PDE[^1] algorithms operating over
> distributed meshes. We have also developed a reference implementation
> of Sieve in C++ as a library of generic algorithms operating on
> distributed containers conforming to the Sieve interface. Sieve makes
> instances of the incidence relation, or *arrows*, the conceptual
> first-class objects represented in the containers. Further, generic
> algorithms acting on this arrow container are systematically used to
> provide natural geometric operations on the topology and also, through
> duality, on the data. Finally, coverings and duality are used to
> encode not only individual meshes, but all types of hierarchies
> underlying PDE data structures, including multigrid and mesh
> partitions.
>
> In order to demonstrate the usefulness of the framework, we show how
> the mesh partition data can be represented and manipulated using the
> same fundamental mechanisms used to represent meshes. We present the
> complete description of an algorithm to encode a mesh partition and
> then distribute a mesh, which is independent of the mesh dimension,
> element shape, or embedding. Moreover, data associated with the mesh
> can be similarly distributed with exactly the same algorithm. The use
> of a high level of abstraction within the Sieve leads to several
> benefits in terms of code reuse, simplicity, and extensibility. We
> discuss these benefits and compare our approach to other existing mesh
> libraries.

# Introduction

Numerical PDE codes frequently comprise of two uneasily coexisting
pieces: the mesh, describing the topology and geometry of the domain,
and the functional *data* attached to the mesh representing the
discretized fields and equations. The mesh data structure typically
reflects the representation used by the mesh generator and carries the
embedded geometic information. While this arrangement is natural from
the point of view of mesh generation and exists in the best of such
packages (e.g., \[17\]), it is frequently foreign to the process of
solving equations on the generated mesh.

At the same time, the functional data closely reflect the linear
algebraic structure of the computational kernels ultimately used to
solve the equations; here the natural geometric structure of the
equations, which reflects the mesh connectivity in the coupling between
the degrees of freedom, is sacrificed to the rigid constraints of the
solver. In particular, the most natural geometric operation of a
restriction of a field to a local neighborhood entails tedious and
error-prone index manipulation.

In response to this state of affairs a number of efforts arose
addressing the fundamental issues of interaction between the topology,
the functional data and algorithms. We note the MOAB project \[20, 19,
8\] and the TSTT/ITAPS SciDAC projects \[8, 16, 3\], the libMesh project
\[6\], the GrAL project \[4\], to name just a few. Sieve shares many
features with these projects, but GrAL is the closest to it in spirit.
Although each of these projects addresses some of the issues outlined
above, we feel that there is room for another approach.

Our Sieve framework, is a collection of interfaces and algorithms for
manipulating geometric data. The design may be summarized by considering
three constructions. First, data in Sieve are indexed by the underlying
geometric elements, such as mesh cells, rather than by some artificial
global order. Further, the local traversal of the data is based on the
connectivity of the geometric elements. For example, Sieve provides
operations that, given a mesh cell, traverse all the data on its
interior, its boundary, or its closure. Typical operations on a Sieve
are shown in Table 1 and described in greater detail in Section 2.1. In
the table, topological mesh elements, such as vertices, edges, and so
on, are refered to as abstract *points* [^2], and the adjacency relation
between two points, such as an edge and its vertex, is refered to as
*covering*: an edge is coverted by its end vertices. Notice that exactly
the same operation is used to obtain edges adjacent to a face as faces
adjacent to a cell, without even a lurking dimension parameter. This is
the *key* to enabling dimension-independent programming for PDE
algorithms.

Second, the global topology is divided into a chain of local topologies
with an overlap structure relating them to each other. The overlap is
encoded using the Sieve data structure again, this time containing
arrows relating points in different local topologies. The data values
over each local piece are manipulated using the local connectivity, and
each local piece may associate different data to the same global
element. The crucial ingredient here is the operation of assembling the
chain of local data collections into a consistent whole over the global
topology.

Third, the covering arrows can carry additional information, controlling
the way in which the data from the covering points are assembled onto
the covered points. For example, orientation information can be encoded
on the arrows to dictate an order for data returned over an element
closure. More sophisticated operations are also possible, such as linear
combinations which enable coordinate transformations, or the projection
and interpolation necessary for multigrid algorithms. This is the
central motivation behind the arrow-centric interface.

Emphasis on the covering idea stems directly from the cell complex
construction in algebraic topology. We have abstracted it along the
lines of category theory, with its emphasis on arrows, or morphisms, as
the organizing principle. The analogy runs deeper, however, because in
PDE applications meshes do not exist for their own sake, but to support
geometrically structured information. The geometric structure of these
data manifests itself through duality between topogical operations, such
as *closure* of a mesh element, and analytical operations, such as the
*restriction* of a field to a closed neighborhood of the element.
Formally this can be seen as a reversal of arrows in a suitable
category. At the practical level, this motivates the arrow-centric point
of view, which allows us to load the arrows with the data (e.g.,
coordinate transformation parameters) making the dualization between
covering and restriction possible.

The arrow-centric point of view also distinguishes our approach from
similar projects such as \[4\]. In addition, it is different from the
concept of a flexible database of geometric entities underlying the MOAB
and TSTT/ITAPS methodologies (see e.g., \[20\] and \[16\]). Sieve can be
thought of as a database, but one that limits the flexibility by
insisting on the arrow-centric structure of the input and output and a
small basic query interface optimized to reveal the covers of individual
elements. This provides a compact conceptual universe shifting the
flexibility to the generic algorithms enabled by a wellcircumscribed
container interface.

Although other compact interfaces based on a similar notion *adjacency*
exist, we feel that Sieve's interface and the notion of a covering
better capture the essense of the geometric notions underlying meshes,
rather than mapping them onto a database-inspired language. Moreover,
these adjacency queries often carry outside information, such as
dimension or shape tags, which is superfluous in the Sieve interface and
limits the opportunity for dimension independent programming. These
geometric notions are so universal that the systematic use of covering
notions is possible at all levels of hierarchy underlying PDE
computation. For example, the notion of covering is used to record
relations between vertices, edges and cells of other dimensions in a
sieve. No separate relation is used to encode "side" adjacencies, such
as "neighbor" relations between cells of the same dimension, as is done
in GrAL.

In fact, the points of a sieve are not a priori interpreted as elements
of different dimensions and covering can be used to encode *overlap*
relations in multiple non-conforming meshes, multigrid hierarchies, or
even identification of cells residing on multiple processors. Contrast
this, for example, with the multiple notions employed by ITAPS to
describe meshes: meshes, submeshes, mesh entities, mesh entity sets and
parallel mesh decompositions. While the relations between all these
concepts are of essentially similar nature, this unity is not apparent
in the interface, inhibiting reuse and hindering analysis of the data
structures, their capabilities and their complexity.

Undoubtedly, other approaches may be more appropriate in other
computational domains. For instance, different data structures may be
more appropriate for mesh generation, where very different types of
queries, modifications and data need to be associated with the mesh.
Partitioning algorithms may also require different data access patterns
to ensure efficiency and scalability. Sieve does not pretend to address
those concerns. Instead, we try to focus on the demands of numerical PDE
algorithms that revolve around the idea of a field defined over a
geometry. Different PDE problems use different fields and even different
numbers of fields with different discretizations. The need for
substantial flexibility in dealing with a broad class of PDE problems
and their geometric nature are the main criterion for the admission into
the Sieve interface.

Here we focus on the reuse of the basic covering notions at different
levels of data hierarchy. In particular, the division of the topology
into pieces and assembly over an overlap is among the fundamental
notions of PDE analysis, numerical or otherwise. It is the essence of
the domain decomposition method and can be used in parallel or serial
settings, or both. Moreover, we focus on this decomposition/assembly
aspect of Sieve and present its capabilities with a fundamental example
of this kind --- the distribution of a mesh onto a collection of
processors. It is a ubiquitous operation in parallel PDE simulation and
a necessary first step in constructing the full distributed problem.
Moreover mesh distribution makes for an excellent pedagogical problem,
illustrating the powerful simplicity of the Sieve construction. The
Sieve interface allows PDE algorithms, operating over data distributed
over a mesh, to be phrased without reference to the dimension, layout,
element shape, or embedding of the mesh. We illustrate this with the
example of distribution of a mesh and associated data fields over it.
The same simple algorithm will be used to distribute an arbitrary mesh,
as well as fields of arbitrary data layout.

We discuss not only the existing code for the Sieve library but also the
concepts that underlie its design and implementation. These two may not
be in complete agreement, as the code continues to evolve. We use the
keyboard font to indicate both existing library interfaces and proposed
developments that more closely relate to our design concepts.
Furthermore, early implementations may not be optimal from the point of
view of runtime and storage complexity as we resist premature
optimizations in favor of refining the interface. Nonetheless, our
reference implementation is fully functional, operating in parallel, and
in use by real applications \[21, 15\]. This implementation verifies the
viability and the consistency of the interface, but does not preclude
more efficient implementations better suited to particular uses. The
added value of the interface comes in the enabling of generic
algorithms, which operate on the interface and are independent of the
underlying implementation. In this publication we illustrate some of
these fundamental algorithms.

The rest of the paper is organized as follows. In Section 2 we introduce
the basic notions and algorithms of the Sieve framework, which are then
seen in action in Section 3 where the algorithms for mesh distribution
and redistribution in a parallel setting are discussed. Section 4
contains specific examples of mesh distribution and Section 5 concludes
the paper.

# Sieve Framework

Sieve can be viewed as a library of parallel containers and algorithms
that extends the standard container collection (e.g., the Standard
Template Library of C++ and BOOST libraries). The extensions are simple
but provide the crucial functionality and introduce what is, in our
view, a very useful semantics. Throughout this paper we freely use the
modern terminology of generic programming, in particular the idea of a
*concept*, which is an interface that a class must implement to be
usable by templated algorithms or methods.

Our fundamental concept is that of a Map, which we understand in the
multivalued sense as an assignment of a sequence of *points* in the
range to each of the points in the domain. A sequence is an immutable
ordered collection of points that can be traversed from the begin
element to the end. Typically a sequence has no repetitions, and we
assume such *set* semantics of sequences unless explicitly noted
otherwise.

A sequence is a basic input and output type of most Sieve operations,
and the basic operation acting on sequences is called restrict. In
particular, a Map can be restricted to a point or a sequence in the
domain, producing the corresponding sequence in the range. Map objects
can be updated in various ways. At the minimum we require that a Map
implement a set operation that assigns a sequence to a given domain
point. Subsequent restrict calls may return a sequence reordered in an
implementation-dependent way.

## Basic containers

Sieve extends the basic Map concept in several ways. First, it allows
bidirectional mappings. Hence we can map points in the range, called the
cap, to the points in the domain, called the base. This mapping is
called the support, while the base-to-cap mapping is called the cone.

Second, the resulting sequence actually contains not the image points
but arrows. An arrow responds to source and target calls, returning
respectively the cap and base points of the arrow. Thus, an arrow not
only abstracts the notion of a pair of points related by the map but
also allows the attachment of nearly arbitrary "payload", a capability
useful for local traversals.

One can picture a Sieve as a bipartite graph with the cap above the base
and the arrows pointing downward (e.g., Fig. 1). The containers are not
constrained by the type of point and arrow objects, so Sieve must be
understood as a library of *meta-objects* and *meta-algorithms* (a
template library in the C++ notation), which generates appropriate code
upon instantiation of basis objects. We primarily have the C++ setting
in mind, although appropriate Python and C bindings have been provided
in our reference implementation.

A Sieve can be made into a Map in two different ways, by identifying
either cone or support with restrict. Each can be done with a simple
adapter class and allows all the basic Map algorithms to be applied to
Sieve objects.

The Sieve also extends Map with capabilities of more geometric
character. It allows the taking of a transitive closure of cone to
obtain the topological closure of a point familiar from cell complex
theory \[10, 1\]. Here arrows are interpreted as the incidence relations
between points, which represent the cells. Likewise, iterated supports
result in the star of a point. The meet(p,q) lattice operation returns
the smallest sequence of points whose removal would render closure(p)
and closure(q) disjoint. The join(p,q) operation is the analogue for
star(p) and star(q). Note that all these operations actually return
arrow sequences, but by default we extract either the source or the
target, a strategy that aids in the definition of transitive closures
and simplifies programming.

Fig. 1 illustrates how mesh topology can be represented as a Sieve
object. The arrows indicate covering or incidence relations between
triangles, edges, and vertices of a simple simplicial mesh. Sieve
operations allow one to navigate through the mesh topology and carry out
the traversals needed to use the mesh. We illustrate some common Sieve
operations on the mesh from Fig. 1 in Table 2.

7

8

6

5

2

3

0

1

4

9

10

2

3

4

5

6

9

8

10

7

1

0

> Figure 1: A simple mesh and its Sieve representation.

  -----------------------------------------------------------------------
  cone(p)         sequence of points covering a given point p
  --------------- -------------------------------------------------------
  closure(p)      transitive closure of cone

  support(p)      sequence of points covered by a given point p

  star(p)         transitive closure of support

  meet(p,q)       minimal separator of closure(p) and closure(q)

  join(p,q)       minimal separator of star(p) and star(q)
  -----------------------------------------------------------------------

Table 1: Typical operations on a Sieve.

+---------------------------+------------------------------------------+
|                           | ![](./seive/media/                       |
|                           | image1.png){width="2.3233333333333333in" |
|                           | height="0.38in"}                         |
+:==========================+:=========================================+
| > closure(1)              | > {4, 5, 6, 7, 10, 8}                    |
+---------------------------+------------------------------------------+
| > star(8)                 | {2, 4, 6, 0, 1}                          |
+---------------------------+------------------------------------------+
|                           | ![](./seive/media/                       |
|                           | image2.png){width="2.3233333333333333in" |
|                           | height="0.5833333333333334in"}           |
+---------------------------+------------------------------------------+

Table 2: Results of typical operations on the Sieve from Fig. 1.

## Data Definition and Assembly

Sieves are designed to represent relations between geometric entities,
represented by points. They can also be used to attach data directly to
arrows, but not to points, since points may be duplicated in different
arrows. A Map, however, can be used effectively to lay out data over
points. It defines a sequence-valued function over the implicitly
defined domain set. In this case the domain carries no geometric
structure, and most data algorithms rely on this minimal Map concept.

### Sections

If a Map is combined with a Sieve, it allows more sophisticated data
traversals such as restrictClosure or restrictStar. These algorithms are
essentially the composition of maps from points to point sets (closure)
with maps from points to data (section). Analogous traversals based on
meet, join, or other geometric information encoded in Sieve can be
implemented in a straightforward manner. The concept resulting from this
combination is called a Section, by analogy with the geometrical notion
of a section of a fiber bundle. Here the Sieve plays the role of the
base space, organizing the points over which the mapping representing
the section is defined. We have found Sections most useful in
implementating finite element discretizations of PDE problems. These
applications of Section functionality are detailed in an upcoming
publication \[14\].

A particular implementation of Map and Section concepts ensures
contiguous storage for the values. We mention it because of its
importance for high-performance parallel computing with Sieve. In this
implementation a Map class uses another Map internally that maps domain
points to offsets into a contiguous storage array. This allows Sieve to
interface with parallel linear and nonlinear solver packages by
identifying Map with the vector from that package. We have done this for
the PETSc \[2\] package. The internal Map is sometimes called the
*atlas* of that Section. The analogous geometric object is the local
trivialization of a fiber bundle that organizes the space of values over
a domain neighborhood (see, e.g., \[18\]).

We observe that Sections and Sieves are in duality. This duality is
expressed by the relation of the restrict operation on a Section to the
cone operation in a Sieve. Corresponding to closure is the traversal of
the Section data implemented by restrictClosure. In this way, to any
Sieve traversal, there corresponds a traversal of the corresponding
Section. Pictured another way, the covering arrows in a Sieve may be
reversed to indicate restriction. This duality will arise again when we
picture the dual of a given mesh in Section 3.1.

### Overlap **and** Delta

In order to ensure efficient local manipulation of the data within a Map
or a Section, the global geometry is divided into manageable pieces,
over which the Maps are defined. In the context of PDE problems, the
chain of subdomains typically represents local meshes that cover the
whole domain. The dual chain, or a cochain, of Maps represents
appropriate restrictions of the data to each subdomain. For PDEs, the
cochain comprises local fields defined over submeshes.

The covering of the domain by subdomains is encoded by an Overlap
object. It can be implemented by a Sieve, whose arrows connect the
points in different subdomains that cover each other. Strictly speaking,
Overlap arrows relate pairs (domain, domain point). Alternatively, we
can view Overlap itself as a chain of Sieves indexed by nonempty
overlaps of the subdomains in the original chain. This better reflects
the locality of likely Overlap traversal patterns: for a given chain
domain, all points and their covers from other subdomains are examined.

An Overlap is a many-to-many relation. In the case of meshes this allows
for nonconforming overlapping submeshes. However, the essential uses of
Overlap are evident even in the simplest case representing conforming
subdomain meshes treated in detail in the example below. Fig. 2
illustrates the Overlap corresponding to a conforming mesh chain
resulting from partitioning of the mesh in Fig. 1. Here the Overlap is
viewed as a chain of Sieves, and the local mesh point indices differ
from the corresponding global indices in Fig. 1. This configuration
emphasizes the fact that no global numbering scheme is imposed across a
chain and the global connectivity is always encoded in the Overlap. In
the present case, this is simply a one-to-one identification relation.
Moreover, many overlap representations are possible; the one presented
above, while straightforward, differs from that shown in Section 3.2.

The values in different Maps of a cochain are related as well. The
relation among them reflects the overlap relation among the points in
the underlying subdomain chain. The nature of the relationship between
values varies according to the problem. For example, for conforming
meshes (e.g., Fig. 2) the Overlap is a one-to-one relation between
identified elements of different subdomain meshes. In this case, the Map
values over the same mesh element in different domains can be
duplicates, as in finite differences, or partial values that have to be
added to obtain the unique global value, as in finite element methods.
In either case the number of values over a shared mesh element must be
the same in the cooverlapping Maps. Sometimes this number is referred to
as the *fiber dimension*, by analogy with fiber bundles.

Vertex coordinates are an example of a cochain whose values are simply
duplicated in different local maps, as shown in Section 3.2. In the case
of nonconforming subdomain meshes, Overlap is a many-to-many relation,
and Map values over overlapping points can be related by a nontrivial
transformation or a relation. They can also be different in number. All
of this information --- fiber dimensions over overlapping points, the
details of the data transformations, and other necessary information ---
is encoded in a

Delta class.

A Delta object can be viewed as a cochain of maps over an Overlap chain,
and is dual to the Overlap in the same way that a Section is dual to a
Sieve.

> ![](./seive/media/image3.png){width="4.5in"
> height="4.633333333333334in"}

Figure 2: Overlap of a conforming mesh chain obtained from breaking up
the mesh in Fig. 1.

More important, a Delta acts on the Map cochain with domains related by
the Overlap. Specifically, the Delta class defines algorithms that
restrict the values from a pair of overlapping subdomains to their
intersection. This fundamental operation borrowed from the *sheaf*
theory (see, e.g., \[5\]) allows us to detect Map cochains that agree on
their overlaps. Moreover (and this is a uniquely computational feature),
Delta allows us to fuse the values on the overlap back into the
corresponding local Maps so as to ensure that they agree on the overlap
and define a valid global map. The restrict-fuse combination is a
ubiquitous operation called completion, which we illustrate here in
detail in the case of *distributed* Overlap and Delta. For example, in
Section 3.2 we use completion to enforce the consistency of cones over
points related by the overlap.

If the domain of the cochain Map carries no topology --- no connectivity
between the points --- it is simply a set and need not be represented by
a Sieve. This is the case for a pure linear algebra object, such as a
PETSc Vec. However, the Overlap and Delta still contain essential
information about the relationship among the subdomains and the data
over them, and must be represented and constructed explicitly. In fact,
a significant part of an implementation of any domain decomposition
problem should be the specification of the Overlap and Delta pair, as
they are at the heart of the problem.

Observe that Overlap fulfills Sieve functions at a larger scale,
encoding the domain topology at the level of subdomains. In fact,
Overlap can be thought of as the "superarrows" between domain
"superpoints." Thus, the essential ideas of encoding topology by arrows
indicating overlap between pieces of the domain is the central idea
behind the Sieve interface. Likewise, Deltas act as Maps on a larger
scale and can be restricted in accordance with an Overlap.

## Database interpretation

The arrow-centric formalism of Sieve and the basic operations have an
interpretations in terms of relational databases and the associated
'entity-relation' analyses. Indeed, Sieve points can naturally be
interpreted as the rows of a table of 'entities' (both in the database
sense and the sense of 'topological entity') with the point itself
serving as the key. Arrows encode covering relations between points, and
therefore define a natural binary database relation with the composite
key consisting of the two involved points. In this scenario cones and
supports have various interpretations in terms of queries against such a
schema; in particular, the cone can be viewed as the result of a
(database) join of the arrow table with the point table on the target
key; the support is the join with the source key. More interestingly,
the topological closure is the transitive closure of the database join
applied to the arrow table; similarly for star. Moreover, meet and join
in the topological sense cannot be formulated quite as succinctly in
terms of database queries, but are very clear in terms of the geometric
intuitive picture of Sieve.

This can be contrasted with the scenario, in which only point entity
tables are present and the covering or incident points are stored in the
entity record alongside the point key. In this case, however, arrows
have no independent existence, are incapable of carrying their own
ancillary information and are duplicated by each of the two related
points. While in this paper we do not focus on the applications of
arrow-specific data that can be attached to the arrow records for lack
of space, we illustrate its utility with a brief sketch of an example.

In extracting the cone or the (topological) closure of a point, such as
a hexahedron in a 3D hex mesh, it is frequently important to traverse
the resulting faces, edges and points in the order determined by the
orientation of the covered hex. Each face, except those on the boundary,
cover two hexahedra and most edges and vertices cover several faces and
edges, respectively. Each of those covering relations induces a
different orientation on the face, edge or vertex. In FEM applications
this results in a change of the sign of integral over the covering
point. The sign, however, is not intrinsically associated with the
covering point, by rather with its orientation relative to the
orientation induced by the covered entity. Thus, the sign of the
integral is determined by the (covering,covered) pair, that is, by the
arrow. In a entity-only schema, at worst there would be no natural place
for the orientation data, and at best it would make for an awkward
design and potentially lead to storage duplication. More sophisticated
uses of arrow-specific data include general transformation of the data
attached to points upon its pullback onto the covered points (consider,
for example, the restriction/prolongation multigrid operators).

To summarize, Sieve can be viewed as an interface defining a relational
database with a very particular schema and a limit query set. This query
set, however, allows for some operations that may be difficult to
describe succinctly in the database language (topological meet and
join)). Furthermore, by defining a *restricted* database of topological
entities and relations, as opposed to a flexible one, Sieve potentially
allows for more effective optimizations of the runtime and storage
performance behind the same interface. These issues will be discussed
elsewhere.

# Mesh Distribution

Before our mesh is distributed, we must decide on a suitable partition,
for which there are many excellent packages (see, e.g., \[12, 13, 11\]).
We first construct suitable overlap Sieves. The points will be abstract
"partitions" that represent the sets of cells in each partition, with
the arrows connecting abstract partitions on different processes. The
Overlap is used to structure the communication of Sieve points among
processes since the algorithm operates only on Sections, in this case we
exhibit the mesh Sieve as a Section with values in the space of points.

## Dual Graph and Partition encoding

The graph partitioning algorithms in most packages, for example ParMetis
and Chaco which were used for testing, require the dual to our original
mesh, sometimes referred to as the element connectivity graph. These
packages partition vertices of a graph, but FEM computations are best
load-balanced by partitioning elements. Consider the simple mesh and its
dual, shown in Fig. 3. The dual Sieve is identical to the original
except that all arrows are reversed. Thus, we have an extremely simple
characterization of the dual.

It is common practice to omit intermediate elements in the Sieve, for
instance storing only cells and vertices. In this case, we may construct
the dual edges on the fly by looping over all cells in the mesh, taking
the support, and placing a dual edge for any support of the correct size
(greater than or equal to the dimension is sufficient) between the two
cells in the support. Note this algorithm also works in parallel because
the supports will, by definition, be identical on all processes after
support completion. Moreover, it is independent of the cell shape and
dimension, unless the dual edges must be constructed.

The partitioner returns an assignment of cells, vertices in the dual, to
partitions. This can be thought of as a Section over the mesh, giving
the partition number for each cell. However, we will instead interpret
this assignment as a Section over the abstract partition points taking
values in

Figure 3: A simple mesh and its dual.

the space of Sieve points, which can be used directly in our generic
Section completion routine, described in Section 3.2.1. In fact, Sieve
can generate a partition of mesh elements of any dimension, for example
mesh faces in a finite volume code, using a hypergraph partitioner, such
as that found in Zoltan \[7\] and exactly the same distribution
algorithm.

## Distributing a Serial Mesh

To make sense of a finite element mesh, we must first introduce a few
new classes. A Topology combines a sequence of Sieves with an Overlap.
Our Mesh is modeled on the fiber bundle abstraction from topology.
Analogous to a topology combined with a fiber space, a Mesh combines a
Topology with a sequence of Sections over this topology. Thus, we may
think of a Mesh as a Topology with several distinguished Sections, the
most obvious being the vertex coordinates.

After the topology has been partitioned, we may distribute the Mesh in
accordance with it, following the steps below:

1.  Distribute the Topology.

2.  Distribute maps associated to the topology.

3.  Distribute bundle sections.

Each distribution is accomplished by forming a specific Section, and
then distributing that Section in accordance with a given overlap. We
call this process *section completion*, and it is responsible for all
communication in the Sieve framework. Thus, we reduce parallel
programming for the Sieve to defining the correct Section and Overlap,
which we discuss below.

### **Section Completion**

Section completion is the process of completing cones, or supports, over
a given overlap. Completion means that the cone over a given point in
the Overlap is sent to the Sieve containing the neighboring point, and
then fused into the existing cone of that neighboring point. By default,
this fusion process is just insertion, but any binary operation is
allowed. For maximum flexibility, this operation is not carried out on
global Sections, but rather on the restriction of a Section to the
Overlap, which we term *overlap sections*. These can then be used to
update the global Section.

The algorithm uses a recursive approach based on our decomposition of a
Section into an atlas and data. First the atlas, also a Section, is
distributed, allowing receive data sizes to be calculated. Then the data
itself is sent. In this algorithm, we refer to the atlas, and its
equivalent for section adapters, as a *sizer*. Here are the steps in the
algorithm:

1.  Create send and receive sizer overlap sections.

2.  Fill send sizer section.

3.  Communicate.

4.  Create send and receive overlap sections.

5.  Fill send section.

6.  Communicate.

The recursion ends when we arrive at a ConstantSection, described in
\[14\], which does not have to be distributed because it has the same
value on every point of the domain.

### **Sieve Construction**

The distribution process uses only section completion to accomplish all
communication and data movement. We use adapters \[9\] to provide a
Section interface to data, such as the partition. The
PartitionSizeSection adapter can be restricted to an abstract partition
point, returning the total number of sieve points in the partition (not
just the those divided by the partitioner). Likewise, the
PartitionSection returns the points in a partition when restricted to
the partition point. When we complete this section, the points are
distributed to the correct processes. All that remains is to establish
the correct hierarchy among these points, which we do by establishing
the correct cone for each point. The ConeSizeSection and ConeSection
adapters for the Sieve return the cone size and points respectively when
restricted to a point. We see here that a sieve itself can be considered
a section taking values in the space of points. Thus sieve completion
consists of the following:

1.  Construct local mesh from partition assignment by copying.

2.  Construct initial partition overlap.

3.  Complete the partition section to distribute the cells.

4.  Update the Overlap with the points from the overlap sections.

5.  Complete the cone section to distribute remaining Sieve points.

6.  Update local Sieves with cones from the overlap sections.

The final Overlap now relates the parallel Sieve to the initial serial
Sieve. Note that we have used only the cone() primitive, and thus this
algorithm applies equally well to meshes of any dimension, element
shape, or connectivity. In fact, we could distribute an arbitrary graph
without changing the algorithm.

## Redistributing a Mesh

Redistributing an existing parallel mesh is identical to distributing a
serial mesh in our framework. However, now the send and receive Overlaps
are potentially nonempty for every process. The construction of the
intermediate partition and cone Sections, as well as the section
completion algorithm, remain exactly as before. Thus, our high level of
abstraction has resulted in enormous savings through code reuse and
reduction in complexity.

As an example, we return to the triangular mesh discussed earlier.
However, we will begin with the distributed mesh shown in Fig. 4, which
assigns triangles (4, 5, 6, 7) to process 0, and (0, 1, 2, 3) to
process 1. The main difference in this example will be the Overlap,
which determines the communication pattern. In Fig. 5, we see that each
process will both send and receive data during the redistribution. Thus,
the partition Section in Fig. 6 has data on both processes. Likewise,
upon completion we can construct a Sieve Overlap with both send and
receive portion on each process. Cone and coordinate completion also
proceed exactly as before, except that data will flow between both
processes. We arrive in the end at the redistributed mesh shown in Fig.
7. No operation other than Section completion itself was necessary.

# Examples

To illustrate the distribution method, we begin with a simple square
triangular mesh, shown in Fig. 8 with its corresponding Sieve shown in
Fig. 9. We distribute this mesh onto two processes: the partitioner
assigns triangles (0, 1, 2, 4) to process 0, and (3, 5, 6, 7) to
process 1. In step 1, we create a local Sieve on process 0, shown in
Fig. 10, since we began with a serial mesh.

For step 2, we identify abstract partition points on the two processes
using an overlap Sieve, shown in Fig. 11. Since this step is crucial to
an understanding of the algorithm, we will explain it in detail. Each
Overlap is a Sieve, with dark circles representing abstract partition
points, and light circles process ranks. The rectangles are Sieve arrow
data, or labels, representing remote partition points. The send Overlap
is shown for process 0, identifying the partition point 1 with the same
point on process 1. The corresponding receive Overlap is shown for
process 1. The send Overlap for process 1 and receive Overlap for
process 0 are both null because we are broadcasting a serial mesh from
process 0.

We now complete the partition Section, using the partition Overlap, in
order to distribute the Sieve points. This Section is shown in Fig. 12.
Not only are the four triangles in partition 1 shown, but also the six
vertices. The receive overlap Section has a base consisting of the
overlap points, in this

> ![](./seive/media/image6.jpg){width="5.0157436570428695in"
> height="5.00001968503937in"}
>
> Figure 4: Initial distributed triangular mesh.

1

0

0

1

0

1

1

1

1

0

0

0

Process 0

Process 1

Figure 5: Partition point Overlap, with dark partition points, light
process ranks, and arrow labels representing remote points. The send
Overlap is on the left, and the receive Overlap on the right.

02891011121718192022

23

0

Process 1

1

32

6711131415162628293031

Process 0

Figure 6: Partition section, with circular partition points and
rectangular Sieve point data.

case partition point 1; the cap will be completed, meaning that it now
has the Sieve points in the cap.

Using the receive overlap Section in step 4, we can update our Overlap
with the new Sieve points just distributed to obtain the Overlap for
Sieve points rather than partition points. The Sieve Overlap is shown in
Fig. 13. Here identified points are the same on both processes, but this
need not be the case. In step 5 we complete the cone Section, shown in
Fig. 14, distributing the covering relation. We use the cones in the
receive overlap Section to construct the distributed Sieve in Fig. 15.

After distributing the topology, we distribute any associated Sections
for the Mesh. In this example, we have only a coordinate Section, shown
in Fig. 16. Notice that while only vertices have coordinate values, the
Sieve Overlap contains the triangular faces as well. Our algorithm is
insensitive to this, as the faces merely have empty cones in this
Section. We now make use of another adapter, the Atlas, which
substitutes the number of values for the values returned by a restrict,
which we use as the sizer for completion. After distribution of this
Section, we have the result in Fig. 17. We are thus able to fully
construct the distributed mesh in Fig. 18.

> ![](./seive/media/image7.jpg){width="5.248456911636046in"
> height="5.225589457567804in"}
>
> Figure 7: Redistributed triangular mesh.

8

9

10

11

12

13

14

15

16

0

1

5

7

4

6

2

3

> Figure 8: A simple triangular mesh.

0

1

2

3

7

4

6

5

12

16

14

15

13

11

10

8

9

> Figure 9: Sieve for mesh in Fig. 8.

0

1

2

4

10

8

9

8

9

11

8

12

11

9

11

14

Process 0

Figure 10: Initial local sieve on process 0 for mesh in Fig. 8.

1

1

1

0

1

1

Process 1

Process 0

Figure 11: Partition point Overlap, with dark partition points, light
process ranks, and arrow labels representing remote points.

3567111213141516

1

Figure 12: Partition section, with circular partition points and
rectangular Sieve point data.

> ![](./seive/media/image8.png){width="4.496666666666667in"
> height="3.5966666666666667in"}

Figure 13: Sieve overlap, with Sieve points in blue, process ranks in
green, and arrow labels representing remote sieve points.

1113

15

151411

1311

12

161513

3

7

6

5

Figure 14: Cone Section, with circular Sieve points and rectangular cone
point data.

> ![](./seive/media/image9.png){width="4.5in"
> height="2.5933333333333333in"}

Figure 15: Distributed Sieve for mesh in Fig. 18.

8

11

10

9

16

12

13

14

15

0.5

0.0

0.0

0.5

1.0

0.5

1.0

1.0

0.0

0.0

0.5

0.0

0.0

1.0

0.5

0.5

1.0

1.0

Figure 16: Coordinate Section, with circular Sieve points and
rectangular coordinate data.

> ![](./seive/media/image10.png){width="4.5in" height="2.93in"}

Figure 17: Distributed coordinate Section.

The mesh distribution method is independent of the topological dimension
of the mesh, the embedding, the cell shapes, and even the type of
element determining the partition. Moreover, it does not depend on the
existence of intermediate mesh elements in the Sieve. We will change
each of these in the next example, distributing a three-dimensional
hexahedral mesh, shown in Fig. 19, by partitioning the faces. As one can
see from Fig. 20, the Sieve is complicated even for this simple mesh.
However, it does have recognizable structures. Notice that it is
stratified by the topological dimension of the points. This is a feature
of any cell complex when represented as a Sieve.

The partition Overlap in this case is exactly the one shown in Fig. 11;
even though an edge partition was used instead of the cell partition
common for finite elements, the partition Section in Fig. 21 looks the
same although with more data. Not only is the closure of the edges
included, but also their star. This is the abstract method to determine
all points in a given partition. The Sieve Overlap after completion is
also much larger but has exactly the same structure. In fact, all
operations have exactly the same form because the section completion
algorithm is independent of all the extraneous details in the problem.
The final partitioned mesh is shown in Fig. 22, where we see that ghost
cells appear automatically when we use a face partition.

> ![](./seive/media/image11.jpg){width="4.168661417322834in"
> height="4.168661417322834in"}
>
> Figure 18: The distributed triangular mesh.

10

9

8

11

20

29

19

28

12

21

30

13

22

31

14

23

32

15

24

33

16

25

34

18

27

17

26

0

1

2

3

4

5

6

7

> Figure 19: A simple hexahedral mesh.
>
> ![](./seive/media/image12.png){width="1.7633333333333334in"
> height="8.013334426946631in"}
>
> Figure 20: Sieve corresponding to the mesh in Fig. 19.

<table style="width:100%;">
<colgroup>
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
<col style="width: 7%" />
</colgroup>
<thead>
<tr>
<th style="text-align: left;"><blockquote>
<p>0</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>1</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>2</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>3</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>5</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>6</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>8</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>9</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>10</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>11</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>12</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>13</p>
</blockquote></th>
<th style="text-align: left;"><blockquote>
<p>15</p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="3" style="text-align: left;"><blockquote>
<p>16 17 18</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>19</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>20</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>21</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>22</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>24</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>25</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>27</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>28</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>29</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>34</p>
</blockquote></td>
</tr>
<tr>
<td colspan="3" style="text-align: left;"><blockquote>
<p>35 36 37</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>38</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>39</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>40</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>41</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>42</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>43</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>45</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>46</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>47</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>48</p>
</blockquote></td>
</tr>
<tr>
<td colspan="3" style="text-align: left;"><blockquote>
<p>49 50 51</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>52</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>53</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>54</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>55</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>56</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>57</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>58</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>59</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>60</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>61</p>
</blockquote></td>
</tr>
<tr>
<td colspan="3" style="text-align: left;"><blockquote>
<p>62 63 64</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>65</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>66</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>67</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>68</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>69</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>70</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>71</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>72</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>73</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>74</p>
</blockquote></td>
</tr>
<tr>
<td colspan="3" style="text-align: left;"><blockquote>
<p>75 76 77</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>78</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>89</p>
</blockquote></td>
<td style="text-align: left;"><blockquote>
<p>96</p>
</blockquote></td>
<td style="text-align: left;">101</td>
<td style="text-align: left;">102</td>
<td style="text-align: left;">103</td>
<td style="text-align: left;">104</td>
<td style="text-align: left;">105</td>
<td style="text-align: left;">107</td>
<td style="text-align: left;">108</td>
</tr>
</tbody>
</table>

109

1

Figure 21: Partition Section, with circular partition points and
rectangular Sieve point data.

# Conclusions

We have presented mesh partitioning and distribution in the context of
the Sieve framework in order to illustrate the power and flexibility of
this approach. Since we draw no distinction between mesh elements of any
shape, dimension, or geometry, we may accept a partition of any element
type, such as cells or faces. Once provided with this partition and an
overlap sieve, which just indicates the flow of information and is
constructed automatically, the entire mesh can be distributed across
processes by using a single operation, *section completion*. Thus, only
a single parallel operation need be portable, verifiable, or optimized
for a given architecture. Moreover, this same operation can be used to
distribute data associated with the mesh, in any arbitrary
configuration, according to the same partition. Thus, the high level of
mathematical abstraction in the Sieve interface results in concrete
benefits in terms of code reuse, simplicity, and extensibility.

# Acknowledgements {#acknowledgements .unnumbered}

The authors benefited from many useful discussions with Gary Miller and
Rob Kirby. This work was supported by the Mathematical, Information, and
Computational Sciences Division subprogram of the Office of Advanced
Scientific Computing Research, Office of Science, U.S. Department of
Energy,

10

9

8

11

20

29

19

28

12

21

13

22

15

24

16

25

34

18

27

17

0

1

2

3

5

6

20

29

19

28

21

30

13

22

31

14

23

32

15

24

33

16

25

34

18

27

17

26

0

3

4

5

6

7

Process 0

Process 1

> Figure 22: Distributed hexahedral mesh. under Contract
> DE-AC02-06CH11357.

# References {#references .unnumbered}

1.  Pavel S. Aleksandrov. *Combinatorial Topology*, volume 3. Dover,
    1998.

2.  Satish Balay, Kris Buschelman, Victor Eijkhout, William D. Gropp,
    Dinesh Kaushik, Matthew G. Knepley, Lois Curfman McInnes, Barry F.
    Smith, and Hong Zhang. PETSc users manual. Technical Report
    ANL95/11 - Revision 3.0.0, Argonne National Laboratory, 2009.

3.  Mark W. Beall, Joe Walsh, and Mark S. Shephard. A comparison of
    techniques for geometry access related to mesh generation.
    *Engineering With Computers*, 20(3):210--221, 2004.

4.  Guntram Berti. *Generic Software Components for Scientific
    Computing*. PhD thesis, TU Cottbus, 2000.
    [http://www.math.tu-cottbus.](http://www.math.tu-cottbus.de/~berti/diss)

> [de/\~berti/diss.](http://www.math.tu-cottbus.de/~berti/diss)

5.  Glen E. Bredon. *Sheaf Theory*. Graduate Texts in Mathematics.

> Springer, 1997.

6.  Graham F. Carey, Michael L. Anderson, Brian R. Carnes, and
    Benjamin S. Kirk. Some aspects of adaptive grid technology related
    to boundary and interior layers. *Journal of Computational Applied
    Mathematics*, 166(1):55--86, 2004.

7.  Karen D. Devine, Erik G. Boman, Robert T. Heaphy, Umit V.¨

> C¸atalyu¨rek, and Robert H. Bisseling. Parallel hypergraph
> partitioning for irregular problems. *SIAM Parallel Processing for
> Scientific Computing*, February 2006.

8.  Ray Meyers et. al. SNL implementation of the TSTT mesh interface. In
    *8th International conference on numerical grid generation in
    computational field simulations*, June 2002.

9.  Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides.
    *Design Patterns*. Addison-Wesley Professional, January 1995.

10. Allen Hatcher. *Algebraic Topology*. Cambridge University
    Press, 2002. \[11\] Bruce Hendrickson and Robert Leland. A
    multilevel algorithm for partitioning graphs. In *Supercomputing
    '95: Proceedings of the 1995 ACM/IEEE Conference on Supercomputing
    (CDROM)*, page 28, New York, 1995. ACM Press.

<!-- -->

12. George Karypis and Vipin Kumar. A parallel algorithm for multilevel
    graph partitioning and sparse matrix ordering. *Journal of Parallel
    and Distributed Computing*, 48:71--85, 1998.

13. George Karypis et al. ParMETIS Web page, 2005. [http://www.cs.
    umn.edu/\~karypis/metis/parmetis.](http://www.cs.umn.edu/~karypis/metis/parmetis)

14. Matthew G. Knepley and Dmitry A. Karpeev. Sieve implementation.
    Technical Report ANL/MCS to appear, Argonne National Laboratory,
    January 2008.

15. Richard C. Martineau and Ray A. Berry. The pressure-corrected ice
    finite element method for compressible flows on unstructured meshes.
    *Journal of Computational Physics*, 198(2):659--685, 2004.

16. E. S. Seol and Mark S. Shephard. A flexible distributed mesh data
    structure to support parallel adaptive analysis. In *Proceedings of
    the 8th US National Congress on Computational Mechanics*, 2005.

17. Jonathan R. Shewchuk. Triangle: Engineering a 2D quality mesh
    generator and Delaunay triangulator. In Ming C. Lin and Dinesh
    Manocha, editors, *Applied Computational Geometry: Towards Geometric
    Engineering*, volume 1148 of *Lecture Notes in Computer Science*,
    pages 203-- 222. Springer-Verlag, May 1996. From the First ACM
    Workshop on Applied Computational Geometry.

18. Norman Steenrod. *The Topology of Fibre Bundles. (PMS-14)*.
    Princeton University Press, April 1999.

19. Timothy J. Tautges. MOAB-SD: Integrated structured and unstructured
    mesh representation. *Engineering With Computers*, 20:286--293,
    2004.

20. Timothy J. Tautges, Ray Meyers, Karl Merkley, Clint Stimpson, and
    Corey Ernst. MOAB: A mesh-oriented database. Technical Report
    SAND2004-1592, Sandia National Laboratories, April 2004.

21. Charles A. Williams, Brad Aagaard, and Matthew G. Knepley.
    Development of software for studying earthquakes across multiple
    spatial and temporal scales by coupling quasi-static and dynamic
    simulations. In *Eos Transactions of the AGU*. American Geophysical
    Union, 2005. Fall Meeting Supplemental, Abstract S53A-1072.

[^1]:
    > Partial differential equation(s).

[^2]: Our *points* correspond to geometric *entities* in some other
    approaches like MOAB or ITAPS

  -----------------------------------------------------------------------
  **0**
  -----------------------------------------------------------------------

  -----------------------------------------------------------------------

**Unstructured Overlapping Mesh Distribution in Parallel**

MATTHEW G. KNEPLEY, Rice University

MICHAEL LANGE, Imperial College London

GERARD GORMAN, Imperial College London

We present a simple mathematical framework and API for parallel mesh and
data distribution, load balancing, and overlap generation. It relies on
viewing the mesh as a Hasse diagram, abstracting away information such
as cell shape, dimension, and coordinates. The high level of abstraction
makes our interface both concise and powerful, as the same algorithm
applies to any representable mesh, such as hybrid meshes, meshes
embedded in higher dimension, and overlapped meshes in parallel. We
present evidence, both theoretical and experimental, that the algorithms
are scalable and efficient. A working implementation can be found in the
latest release of the PETSc libraries.

Categories and Subject Descriptors: G.4 \[**Mathematical Software**\]:
*Parallel and vector implementations*;

G.1.8 \[**Numerical Analysis**\]: Partial Differential
Equations---*Finite Element Methods*

General Terms: Algorithms, Design, Performance

Additional Key Words and Phrases: mesh distribution, mesh overlap, Hasse
diagram, CW complex, PETSc

**ACM Reference Format:**

Matthew G. Knepley, Michael Lange, and Gerard J. Gorman, 2014.
Unstructured Overlapping Mesh Distribution in Parallel. ACM Trans. Math.
Softw. 0, 0, Article 0 ( 2014), 14 pages. DOI = 10.1145/0000000.0000000
http://doi.acm.org/10.1145/0000000.0000000

arXiv:1506.06194v1 \[cs.MS\] 20 Jun 2015

# INTRODUCTION

The algorithms and implementation for scalable mesh management,
encompassing partitioning, distribution, rebalancing, and overlap
generation, as well as data management over a mesh can be quite complex.
It is common to divide meshes into collections of entities (cell, face,
edge, vertex) of different dimensions which can take a wide variety of
forms (triangle, pentagon, tetrahedron, pyramid, . . . ), and have query

MGK acknowledges partial support from DOE Contract DE-AC02-06CH11357 and
NSF Grant OCI1147680. ML and GJG acknowledge support from EPSRC grant
EP/L000407/1 and the embedded CSE programme of the ARCHER UK National
Supercomputing Service (http://www.archer.ac.uk). All authors
acknowledge support from the Intel Parallel Computing Center program
through grants to both the University of Chicago and Imperial College
London.

Authors' addresses: M.G. Knepley, Computational and Applied Mathematics,
Rice University, Houston, TX; email: knepley@rice.edu; M. Lange,
Imperial College London; email: michael.lange@imperial.ac.uk; G.J.
Gorman, Imperial College London; email: g.gorman@imperial.ac.uk

Permission to make digital or hard copies of part or all of this work
for personal or classroom use is granted without fee provided that
copies are not made or distributed for profit or commercial advantage
and that copies show this notice on the first page or initial screen of
a display along with the full citation. Copyrights for components of
this work owned by others than ACM must be honored. Abstracting with
credit is permitted. To copy otherwise, to republish, to post on
servers, to redistribute to lists, or to use any component of this work
in other works requires prior specific permission and/or a fee.
Permissions may be requested from Publications Dept., ACM, Inc., 2 Penn
Plaza, Suite 701, New York, NY 10121-0701 USA, fax +1 (212) 869-0481, or
permissions@acm.org.

![](./mesh2/media/image1.png){width="0.11in"
height="0.10666666666666667in"}2014 ACM 0098-3500/2014/-ART0 \$10.00

DOI 10.1145/0000000.0000000 http://doi.acm.org/10.1145/0000000.0000000

ACM Transactions on Mathematical Software, Vol. 0, No. 0, Article 0,
Publication date: 2014.

functions tailored to each specific form \[D'Azevedo and et. al. 2015\].
This code structure, however, results in many different cases, little
reuse, and greatly increases the complexity and maintenance burden. On
the other hand, codes for adaptive redistribution of meshes based on
parallel partitioning such as the Zoltan library \[Devine et al. 2006\],
usually represent the mesh purely as an undirected graph, encoding cells
and vertices and ignoring the topology. For data distribution,
interfaces have been specialized to each specific function space
represented on the mesh. In Zoltan, for example, the user is responsible
for supplying functions to pack and unpack data from communication
buffers. This process can be automated however, as in DUNE-FEM \[Dedner
et al. 2010\] which attaches data to entities, much like our mesh points
described below.

We have previously presented a mesh representation which has a single
entity type, called *points*, and a single antisymmetric relation,
called *covering* \[Knepley and Karpeev 2009\]. This structure, more
precisely a Hasse diagram \[Birkhoff 1967; Wikipedia 2015b\], can
represent any CW-complex \[Hatcher 2002; Wikipedia 2015a\], and can be
represented algorithmically as a directed acyclic graph (DAG) over the
points. It comes with two simple relational operations, cone(*p*),
called the *cone* of *p* or the in-edges of point *p* in the DAG, and
its dual operation supp(*p*), called the *support* of *p* or the
out-edges of point *p*. In addition, we will add the transitive closure
in the DAG of these two operations, respectively the closure cl(*p*) and
star st(*p*) of point *p*. In Fig. 1, we show an example mesh and its
corresponding DAG, for which we have cone(*A*) = {*a,b,e*} and supp(*β*)
= {*a,c,e*}, and the transitive closures cl(*A*) = {*A,a,b,e,α,β,γ*} and
st(*β*) = {*β,a,c,e,A,B*}.

> *γ*

*a αδ*

*A*

*B*

*b*

*e*

*c*

*d*

*α*

*β*

*γ*

*δ*

*A*

*B*

*β*

*a*

*b*

*c*

*d*

*e*

Fig. 1. A simplicial doublet mesh and its DAG (Hasse diagram).

In our prior work \[Knepley and Karpeev 2009\], it was unclear whether
simple generic algorithms for *parallel* mesh management tasks could be
formulated, or various types of meshes would require special purpose
code despite the generic mesh representation. Below, we present a
complete set of generic algorithms, operating on our generic DAG
representation, for parallel mesh operations, including partitioning,
distribution, rebalancing, and overlap generation. The theoretical
underpinnings and algorithms are laid out in Section 2, and experimental
results detailed in Section 3.

# THEORY

## Overlap Creation

We will use the Hasse diagram representation of our computational mesh
\[Knepley and Karpeev 2009\], the DMPlex class in PETSc \[Balay et al.
2014a; Balay et al. 2014b\], and describe mesh relations (adjacencies)
with basic graph operations on a DAG. A distributed mesh is a collection
of closed serial meshes, meaning that they contain the closure of each
point, together with an "overlap structure", which marks a subset of the
mesh points and indicates processes with which these points are shared.
The default PETSc representation of the overlap information uses the SF
class, short for *Star Forest* \[Brown 2011\]. Each process stores the
true owner (root) of its own ghost points (leaves), one side of the
relation above, and construct the other side automatically.

In order to reason about potential parallel mesh algorithms, we will
characterize the contents of the overlap using the mesh operations.
These operations will be understood to operate on the entire parallel
mesh, identifying shared points, rather than just the local meshes on
each process. To indicate a purely local operation, we will use a
subscript, e.g. cl~loc~(*p*) to indicate the closure of a point *p*
evaluated only on the local submesh.

The mesh overlap contains all points of the local mesh adjacent to
points of remote meshes in the complete DAG for the parallel mesh, and
we will indicate that point *p* is in the overlap using an indicator
function O. Moreover, if the overlap contains a point *p* on a given
process, then it will also contain the closure of *p*,

O(*p*) =⇒ O(*q*) ∀*q* ∈ cl(*p*)*,* (1)

which shows that if a point is shared, its closure is also shared. This
is a consequence of each local mesh being closed, the transitive closure
of its Hasse diagram. We can now examine the effect of increasing the
mesh overlap in parallel by including all the immediately adjacent mesh
points to each local mesh.

The set of adjacent mesh point differs depending on the discretization.
For example, the finite element method couples unknowns to all other
unknowns whose associated basis functions overlap the support of the
given basis function. If functions are supported on cells whose closure
contains the associated mesh point, we have the relation

adj(*p,q*) ⇐⇒ *q* ∈ cl(st(*p*))*,* (2)

where we note that this relation is symmetric. For example, a degree of
freedom (dof) associated with a vertex is adjacent to all dofs on the
cells containing that vertex. We will call this *FE adjacency*. On the
other hand, for finite volume methods, we typically couple cell unknowns
only through faces, so that we have

adj(*p,q*) ⇐⇒ *q* ∈ supp(cone(*p*))*,* (3)

which is the common notion of cell-adjacency in meshes, and what we will
call *FV adjacency*. This will also be the adjacency pattern for
Discontinuous Galerkin methods.

If we first consider FV adjacency, we see that the cone operation can be
satisfied locally since local meshes are closed. Thus the support from
neighboring processes is needed for all points in the overlap. Moreover,
in order to preserve the closure property of local meshes, the closure
of that support would also need to be collected.

For FE adjacency, each process begins by collecting the star of its
overlap region in the local mesh, st~loc~(O). The union across all
processes will produce the star of each point in the overlap region.
First, note that if the star of a point *p* on the local processes
contains a point *q* on the remote process, then *q* must be contained
in the star of a point *o* in the overlap,

*q* ∈ st(*p*) ⇐⇒ ∃*o* \| O(*o*) ∧ *q* ∈ st(*o*)*.* (4)

There is a path from *p* to *q* in the mesh DAG, since *q* lies in star
of *p*, which is the transitive closure. There must be an edge in this
path which connects a point on the local mesh to one on the remote mesh,
otherwise the path is completely contained in the local mesh. One of the
endpoints *o* of this edge will be contained in the overlap, since it
contains all local points adjacent to remote points in the DAG. In fact,
*q* lies in the star of *o*, since *o* lies on the path from *p* to *q*.
Thus, the star of *p* is contained in the union of the star of the
overlap,

st![](./mesh2/media/image2.png){width="0.8133333333333334in"
height="0.3in"}*.* (5)

Taking the closure of this star is a local operation, since local meshes
are closed. Therefore, parallel overlap creation can be accomplished by
the following sequence: each local mesh collects the closure of the star
of its overlap, communicates this to its overlap neighbors, and then
each neighbor augments its overlap with the new points. Moreover, no
extra points are communicated, since each communicated point *q* is
adjacent to some *p* on a remote process.

## Data Distribution

We will recognize three basic objects describing a parallel data layout:
the Section \[Balay et al. 2014a\] describing an irregular array of data
and the SF, StarForest \[Brown 2011\], a one-sided description of shared
data. A Section is a map from a domain of *points* to data sizes, or
*ndofs*, and assuming the data is packed it can also calculate an offset
for each point. This is exactly the encoding strategy used in the
Compressed Sparse Row matrix format \[Balay et al. 2014a\]. An SF stores
the owner for any piece of shared data which is not owned by the given
process, so it is a onesided description of sharing. This admits a very
sparse storage scheme, and a scalable algorithm for assembly of the
communication topology \[Hoefler et al. 2010\]. The third local object,
a Label, is merely a one-to-many map between integers, that can be
manipulated in a very similar fashion to a Section since the structure
is so similar, but has better complexity for mutation operations.

A Section may be stored as a simple list of (*ndof*, *offset*) pairs,
and the SF as (*ldof*, *rdof*, *rank*) triples where *ldof* is the local
dof number and *rdof* is the remote dof number, which means we never
need a global numbering of the unknowns. Starting with these two simple
objects, we may mechanically build complex, parallel data distributions
from simple algebraic combination operations. We will illustrate this
process with a simple example.

Suppose we begin with a parallel cell-vertex mesh having degrees of
freedom on the vertices. On each process, a Section holds the number of
dofs on each vertex, and **Point Space Dof Space** ~Section~ SF

Solution Dofs Adjacent Dofs Jacobian Layout Shared Adjacency

Mesh Points Solution Dofs Data Layout Shared Dofs

Mesh Points Mesh Points Topology Shared Topology

Processes Mesh Points Point Partition Shared Points

Processes Neighbors

Fig. 2. This figure illustrates the relation between different
Section/SF pairs. The first column gives the domain space for the
Section, the second the range space for the Section and domain and range
for the SF. The Section and SF columns give the semantic content for
those structures at each level, and the arrows show how the SF at each
level can be constructed with input from below. Each horizontal line
describes the parallel layout of a certain data set. For example, the
second line down describes the parallel layout of the solution field.

an SF lists the vertices which are owned by other processes. Notice that
the domain (point space) of the Section is both the domain and the range
(dof space) of the SF. We can combine these two to create a new SF whose
domain and range (dof space) match the range space of the Section. This
uses the PetscSFCreateSectionSF() function, which is completely local
except for the communication of remote dof offsets, which needs a single
sparse broadcast from dof owners (roots) to dof sharers (leaves),
accomplished using PetscSFBcast(). The resulting SF describes the shared
dofs rather than the shared vertices. We can think of the new SF as the
push-forward along the Section map. This process can be repeated to
generate a tower of relations, as illustrated in Fig. 2.

We can illustrate the data structures and transformations in Fig. 2 by
giving concrete examples for the parallel mesh in Fig. 3. Given the
partition in the figure, we have an

SF *SF*~point~, called *Shared Points* in Fig. 2,

![](./mesh2/media/image3.png){width="2.69in" height="0.18in"}*,*

![](./mesh2/media/image4.png){width="2.68in" height="0.18in"}*,*

where the superscript denotes the process on which the object lives. Let
us define a data layout for the solution to a Stokes problem using the
Taylor-Hood \[Taylor and Hood 1973\] finite element scheme
(*P*~2~--*P*~1~). We define the Section *S~u~*, called *Data Layout* in
Fig. 2,

> ![](./mesh2/media/image5.png){width="3.8366666666666664in"
> height="0.38333333333333336in"}*.*

Using PetscSFCreateSectionSF(), we obtain a Section *SF*~dof~, called
*Shared Dof* in

Fig. 2, giving us the shared dofs between partitions,

*SF*~dof~^0^ = {4 → (4*,*1)*,*5 → (5*,*1)*,*6 → (9*,*1)*,*7 →
(10*,*1)*,*8 → (11*,*1)*,*

> 12 → (12*,*1)*,*13 → (13*,*1)*,*14 → (14*,*1)}

which we note is only half of the relation, and SF stores one-sided
information. The other half which is constructed on the fly is

> *SF*~dof~^1^ = {4 → (4*,*0)*,*5 → (5*,*0)*,*9 → (6*,*0)*,*10 →
> (7*,*0)*,*11 → (8*,*0)*,* 12 → (12*,*0)*,*13 → (13*,*0)*,*14 →
> (14*,*0)}*.*

*A*

*B*

*α*

*β*

*γ*

*φ*

*a*

*b*

*δ*

*c*

*d*

*e*

*f*

Fig. 3. A parallel simplicial doublet mesh, with points on process 0
blue and process 1 green.

We can use these same relations to transform any parallel data layout
into another given an SF which connects the source and target point
layouts. Suppose that we have an SF which maps currently owned points to
processes which will own them after redistribution, which we will call a
*migration* SF. With this SF, we can construct the section after
redistribution and migrate the data itself. This process is show in Alg.
1, which uses PetscSFCreateSectionSF() from above to transform the
migration SF over points to one over dofs, and also
PetscSFDistributeSection() to create the section after redistribution.
The section itself can be distributed using only one sparse broadcast,
although we typically use another to setup remote dof offsets for
PetscSFCreateSectionSF(), as shown in Alg. 2.

**Algorithm 1** Algorithm for migrating data in parallel

> 1: **function** MIGRATEDATA(sf, secSource, dtype, dataSource,
> secTarget, dataTarget)

2: PETSCSFDISTRIBUTESECTION(sf, secSource, remoteOff, secTarget)

3: PETSCSFCREATESECTIONSF(sf, secSource, remoteOff, secTarget, sfDof)

4: PETSCSFBCAST(sfDof, dtype, dataSource, dataTarget)

These simple building blocks can now be used to migrate all the data for
a DMPlex object, representing an unstructured mesh of arbitrary
dimension composed of cells, each of which may have any shape. The
migration of cone data, coordinates, and labels all follow the general
migration algorithm above, since each piece of data can be expressed as
the combination of a Section, giving the layout, and an array storing
the values, in PETSc a Vec or IS object. Small differences from the
generic algorithm arise due to **Algorithm 2** Algorithm for migrating a
Section in parallel

> 1: **function** DISTRIBUTESECTION(sf, secSource, remoteOff, secTarget)

2: \<Calculate domain (chart) from local SF points\>

3: PETSCSFBCAST(sf, secSource.dof, secTarget.dof) *.* Move point dof
sizes

> 4: PETSCSFBCAST(sf, secSource.off, remoteOff) *.* Move point dof
> offsets 5: PETSCSECTIONSETUP(secTarget)

the nature of the stored data. For example, the cone data must also be
transformed from original local numbering to the new local numbering,
which we accomplish by first moving to a global numbering and then to
the new local numbering using two local-to-global renumberings. After
moving the data, we can compute a new point SF using Alg. 4, which uses
a reduction to compute the unique owners of all points.

**Algorithm 3** Algorithm for migrating a mesh in parallel

> 1: **function** MIGRATE(dmSource, sf, dmTarget)

2: ISLOCALTOGLOBALMAPPINGAPPLYBLOCK(l2g, csize, cones, cones)

3: *.* Convert to global numbering

4: PETSCSFBCAST(sf, l2g, l2gMig) *.* Redistribute renumbering

5: DMPLEXDISTRIBUTECONES(dmSource, sf, l2gMig, dmTarget)

> 6: DMPLEXDISTRIBUTECOORDINATES(dmSource, sf, dmTarget) 7:
> DMPLEXDISTRIBUTELABELS(dmSource, sf, dmTarget)

**Algorithm 4** Algorithm for migrating an SF in parallel

+----+-----------------------------------------------------------------+
| >  |                                                                 |
| 1: |                                                                 |
| >  |                                                                 |
| ** |                                                                 |
| fu |                                                                 |
| nc |                                                                 |
| ti |                                                                 |
| on |                                                                 |
| ** |                                                                 |
| >  |                                                                 |
|  M |                                                                 |
| IG |                                                                 |
| RA |                                                                 |
| TE |                                                                 |
| SF |                                                                 |
| (s |                                                                 |
| fS |                                                                 |
| ou |                                                                 |
| rc |                                                                 |
| e, |                                                                 |
| >  |                                                                 |
| sf |                                                                 |
| Mi |                                                                 |
| g, |                                                                 |
| >  |                                                                 |
|  s |                                                                 |
| fT |                                                                 |
| ar |                                                                 |
| ge |                                                                 |
| t) |                                                                 |
| >  |                                                                 |
| 2: |                                                                 |
| >  |                                                                 |
| PE |                                                                 |
| TS |                                                                 |
| CS |                                                                 |
| FG |                                                                 |
| ET |                                                                 |
| GR |                                                                 |
| AP |                                                                 |
| H( |                                                                 |
| sf |                                                                 |
| Mi |                                                                 |
| g, |                                                                 |
| >  |                                                                 |
|  N |                                                                 |
| r, |                                                                 |
| >  |                                                                 |
|  N |                                                                 |
| l, |                                                                 |
| >  |                                                                 |
|  l |                                                                 |
| ea |                                                                 |
| ve |                                                                 |
| s, |                                                                 |
| >  |                                                                 |
|  N |                                                                 |
| UL |                                                                 |
| L) |                                                                 |
+:===+:================================================================+
| >  | **for** *p* ← 0*,Nl* **do** *.* Make bid to own all points we   |
| 3: | received                                                        |
+----+-----------------------------------------------------------------+
| >  | > lowners\[p\].rank = rank                                      |
| 4: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > lowners\[p\].index = leaves ? leaves\[p\] : p                 |
| 5: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | **for** *p* ← 0*,Nr* **do** *.* Flag so that MAXLOC does not    |
| 6: | use root value                                                  |
+----+-----------------------------------------------------------------+
| >  | > rowners\[p\].rank = -1                                        |
| 7: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > rowners\[p\].index = -1                                       |
| 8: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > PETSCSFREDUCE(sfMigration, lowners, rowners, MAXLOC)          |
| 9: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > PETSCSFBCAST(sfMigration, rowners, lowners)                   |
|  1 |                                                                 |
| 0: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > **for** *p* ← 0*,Nl,Ng* = 0 **do**                            |
|  1 |                                                                 |
| 1: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > **if** lowners\[p\].rank != rank **then**                     |
|  1 |                                                                 |
| 2: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > ghostPoints\[Ng\] = leaves ? leaves\[p\] : p                  |
|  1 |                                                                 |
| 3: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > remotePoints\[Ng\].rank = lowners\[p\].rank                   |
|  1 |                                                                 |
| 4: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > remotePoints\[Ng\].index = lowners\[p\].index                 |
|  1 |                                                                 |
| 5: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > Ng++                                                          |
|  1 |                                                                 |
| 6: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > PETSCSFSETGRAPH(sfTarget, Np, Ng, ghostPoints, remotePoints)  |
|  1 |                                                                 |
| 7: |                                                                 |
+----+-----------------------------------------------------------------+

## Mesh Distribution

Using the data migration routines above, we can easily accomplish
sophisticated mesh manipulation in PETSc. Thus, we can redistribute a
given mesh in parallel, a special case of which is distribution of a
serial mesh to a set of processes. As shown in Alg. 5, we first create a
partition using a third party mesh partitioner, and store it as a label,
where the target ranks are label values. We take the closure of this
partition in the DAG, invert the partition to get receiver data,
allowing us to create a migration SF and use the data migration
algorithms above. The only piece of data that we need in order to begin,
or bootstrap, the partition process is an SF which connects sending and
receiving processes. Below, we create the complete graph on processes,
meaning that any process could communicate with any other, in order to
avoid communication to discover which processes receive from the
partition. Discovery is possible and sometimes desirable, and will be
incorporated in a further update.

**Algorithm 5** Algorithm for distributing a mesh in parallel

+----+-----------------------------------------------------------------+
| >  |                                                                 |
| 1: |                                                                 |
| >  |                                                                 |
| ** |                                                                 |
| fu |                                                                 |
| nc |                                                                 |
| ti |                                                                 |
| on |                                                                 |
| ** |                                                                 |
| >  |                                                                 |
| DI |                                                                 |
| ST |                                                                 |
| RI |                                                                 |
| BU |                                                                 |
| TE |                                                                 |
| (d |                                                                 |
| m, |                                                                 |
| >  |                                                                 |
| ov |                                                                 |
| er |                                                                 |
| la |                                                                 |
| p, |                                                                 |
| >  |                                                                 |
|  s |                                                                 |
| f, |                                                                 |
| >  |                                                                 |
| pd |                                                                 |
| m) |                                                                 |
|    |                                                                 |
| 2: |                                                                 |
| P  |                                                                 |
| ET |                                                                 |
| SC |                                                                 |
| PA |                                                                 |
| RT |                                                                 |
| IT |                                                                 |
| IO |                                                                 |
| NE |                                                                 |
| RP |                                                                 |
| AR |                                                                 |
| TI |                                                                 |
| TI |                                                                 |
| ON |                                                                 |
| (p |                                                                 |
| ar |                                                                 |
| t, |                                                                 |
| d  |                                                                 |
| m, |                                                                 |
| lb |                                                                 |
| lP |                                                                 |
| ar |                                                                 |
| t) |                                                                 |
| *  |                                                                 |
| .* |                                                                 |
| P  |                                                                 |
| ar |                                                                 |
| ti |                                                                 |
| ti |                                                                 |
| on |                                                                 |
| c  |                                                                 |
| el |                                                                 |
| ls |                                                                 |
+:===+:================================================================+
| >  | DMPLEXPARTITIONLABELCLOSURE(dm, lblPart) *.* Partition points   |
| 3: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | **for** *p* ← 0*,P* **do** *.* Create process SF                |
| 4: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > remoteProc\[p\].rank = p                                      |
| 5: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > remoteProc\[p\].index = rank                                  |
| 6: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > PETSCSFSETGRAPH(sfProc, P, P, NULL, remoteProc)               |
| 7: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > DMPLEXPARTITIONLABELINVERT(dm, lblPart, sfProc, lblMig)       |
| 8: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | *.* Convert from senders to receivers                           |
| 9: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > DMPLEXPARTITIONLABELCREATESF(dm, lblMig, sfMig)               |
|  1 |                                                                 |
| 0: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | *.* Create migration SF                                         |
|  1 |                                                                 |
| 1: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | DMPLEXMIGRATE(dm, sfMigration, dmParallel) *.* Distribute DM    |
|  1 |                                                                 |
| 2: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | DMPLEXDISTRIBUTESF(dm, sfMigration, dmParallel) *.* Create new  |
|  1 | SF                                                              |
| 3: |                                                                 |
+----+-----------------------------------------------------------------+

We can illustrate the migration process by showing how Fig. 3 is derived
from Fig. 1. We begin with the doublet mesh contained entirely on one
process. In the partition phase, we first create a cell partition
consisting of a Section *S*~cpart~ for data layout and an IS cpart
holding the points in each partition,

> *S*~cpart~ = {0 : (1*,*0)*,*1 : (1*,*1)}*,* cpart = {*B,A*}*,*

which is converted to the equivalent Label, a data structure better
optimized for overlap insertion,

*L*~cpart~ = {0 → {*B*}*,*1 → {*A*}}*,*

and then we create the transitive closure. We can express this as a
Section *S*~part~, called *Point Partition* in Fig. 2, and IS part with
the partition data,

> *S*~part~ = {0 : (4*,*0)*,*1 : (7*,*4)}*,* part =
> {*B,c,d,δ,A,a,b,e,α,β,γ*}*,*

or as the equivalent Label

*L*~part~ = {0 → {*B,c,d,δ*}*,*1 → {*A,a,b,e,α,β,γ*}}*.*

The bootstrap SF *SF*~proc~, called *Neighbors* in Fig. 2, encapsulates
the data flow for migration

*SF*~proc~ = {0 → (0*,*0)*,*1 → (1*,*1)}*.*

We have a small problem in that the partition structure specifies the
send information, and for an SF we require the receiver to specify the
data to be received. Thus we need to invert the partition. This is
accomplished with a single call to DMPlexDistributeData() from Alg. 1,
which is shown in Alg. 6. This creates a Section and IS with the receive
information,

> *S*invpart0 = {0 : (4*,*0)} invpart = {*B,c,d,δ*}
>
> *S*invpart1 = {0 : (7*,*0)} invpart = {*A,a,b,e,α,β,γ*}*.*

and then we convert them back into a Label *L*~invpart~. This simple
implementation for the complex operation of partition inversion shows
the power of our flexible interface for data movement. Since the
functions operate on generic representations of data (e.g. Section, SF),
the same code is reused for many different mesh types and mesh/data
operations, and only a small codebase needs to be maintained. In fact,
the distribution (one-to-many) and redistribution (many-to-many)
operations are identical except for an initial inversion of the point
numbering to obtain globally unique numbers for cones.

**Algorithm 6** Algorithm for inverting a partition

> 1: MIGRATEDATA(*SF*~proc~, *S*~part~, MPIU_2INT, part, *S*~invpart~,
> invpart)

After inverting our partition, we combine *L*~invpart~ and *SF*~proc~
using

DMPlexPartitionLabelCreateSF(), the equivalent of
PetscSFCreateSectionSF(), to obtain the SF for point migration

> *SF*~point~ = {*A* → (*A,*1)*,B* → (*B,*0)*, a* → (*a,*1)*,b* →
> (*b,*1)*,c* → (*c,*0)*,d* → (*d,*0)*,e* → (*e,*1)*, α* → (*α,*1)*,β* →
> (*β,*1)*,γ* → (*γ,*1)*,δ* → (*δ,*1)}*.*

In the final step, this SF is then used to migrate all the (Section,
array) pairs in the

DMPlex, such as cones, coordinates, and labels, using the generic
DMPlexMigrate() function.

## Overlap Generation

Following the initial distribution of the mesh, which was solely based
on the partitioner output, the set of overlapping local meshes can now
be derived in parallel. This derivation is performed by each process
computing it's local contribution to the set of overlap points on
neighboring processes, starting from an SF that contains the initial
point sharing. It is important to note here that this approach performs
the potentially costly adjacency search in parallel and that the search
space is limited to the set of points initially shared along the
partition boundary.

The algorithm for identifying the set of local point contributions to
neighboring partitions is based on the respective adjacency definitions
given in section 2.1. As illustrated in Alg. 7, the SF containing the
initial point overlap is first used to identify connections between
local points and remote processes. To add a level of adjacent points,
the local points adjacent to each connecting point are added to a
partition label similar to the one used during the initial migration
(see Alg. 5), identifying them as now also connected to the neighboring
process. Once the point donations for the first level of cell overlap
are defined, further levels can be added through repeatedly finding
points adjacent to the current donations.

**Algorithm 7** Algorithm for computing the partition overlap

+----+-----------------------------------------------------------------+
| >  |                                                                 |
| 1: |                                                                 |
| >  |                                                                 |
| ** |                                                                 |
| fu |                                                                 |
| nc |                                                                 |
| ti |                                                                 |
| on |                                                                 |
| ** |                                                                 |
| >  |                                                                 |
|  D |                                                                 |
| MP |                                                                 |
| LE |                                                                 |
| XC |                                                                 |
| RE |                                                                 |
| AT |                                                                 |
| EO |                                                                 |
| VE |                                                                 |
| RL |                                                                 |
| AP |                                                                 |
| (d |                                                                 |
| m, |                                                                 |
| >  |                                                                 |
| ov |                                                                 |
| er |                                                                 |
| la |                                                                 |
| p, |                                                                 |
| >  |                                                                 |
|  s |                                                                 |
| f, |                                                                 |
| >  |                                                                 |
| od |                                                                 |
| m) |                                                                 |
| >  |                                                                 |
| >  |                                                                 |
| 2: |                                                                 |
| >  |                                                                 |
|  D |                                                                 |
| MP |                                                                 |
| LE |                                                                 |
| XD |                                                                 |
| IS |                                                                 |
| TR |                                                                 |
| IB |                                                                 |
| UT |                                                                 |
| EO |                                                                 |
| WN |                                                                 |
| ER |                                                                 |
| SH |                                                                 |
| IP |                                                                 |
| (d |                                                                 |
| m, |                                                                 |
| >  |                                                                 |
|  s |                                                                 |
| f, |                                                                 |
| >  |                                                                 |
| ro |                                                                 |
| ot |                                                                 |
| Se |                                                                 |
| ct |                                                                 |
| io |                                                                 |
| n, |                                                                 |
| >  |                                                                 |
|  r |                                                                 |
| oo |                                                                 |
| tR |                                                                 |
| an |                                                                 |
| k) |                                                                 |
| >  |                                                                 |
|  * |                                                                 |
| .* |                                                                 |
| >  |                                                                 |
| De |                                                                 |
| ri |                                                                 |
| ve |                                                                 |
| >  |                                                                 |
| se |                                                                 |
| nd |                                                                 |
| er |                                                                 |
| >  |                                                                 |
|  i |                                                                 |
| nf |                                                                 |
| or |                                                                 |
| ma |                                                                 |
| ti |                                                                 |
| on |                                                                 |
| >  |                                                                 |
| fr |                                                                 |
| om |                                                                 |
| >  |                                                                 |
| SF |                                                                 |
|    |                                                                 |
| 3: |                                                                 |
| *  |                                                                 |
| *f |                                                                 |
| or |                                                                 |
| ** |                                                                 |
| *l |                                                                 |
| ea |                                                                 |
| f* |                                                                 |
| ←  |                                                                 |
| *  |                                                                 |
| sf |                                                                 |
| .l |                                                                 |
| ea |                                                                 |
| ve |                                                                 |
| s* |                                                                 |
| ** |                                                                 |
| do |                                                                 |
| ** |                                                                 |
| *  |                                                                 |
| .* |                                                                 |
| A  |                                                                 |
| dd |                                                                 |
| l  |                                                                 |
| oc |                                                                 |
| al |                                                                 |
| r  |                                                                 |
| ec |                                                                 |
| ei |                                                                 |
| ve |                                                                 |
| c  |                                                                 |
| on |                                                                 |
| ne |                                                                 |
| ct |                                                                 |
| io |                                                                 |
| ns |                                                                 |
+:===+:================================================================+
| >  | > DMPLEXGETADJACENCY(sf, leaf.index, adjacency)                 |
| 4: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > **for** *a* ← *adjacency* **do**                              |
| 5: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > DMLABELSETVALUE(lblOl, a, leaf.rank)                          |
| 6: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | **for** *p* ← 0*,P* **do** *.* Add local send connections       |
| 7: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > **if** rootSection\[p\] \> 0 **then**                         |
| 8: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > DMPLEXGETADJACENCY(sf, p, adjacency)                          |
| 9: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > **for** *a* ← *adjacency* **do**                              |
|  1 |                                                                 |
| 0: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > DMLABELSETVALUE(lblOl, a, rootRank\[p\])                      |
|  1 |                                                                 |
| 1: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | **for** *n* ← 1*,overlap* **do** *.* Add further levels of      |
|  1 | adjacency                                                       |
| 2: |                                                                 |
+----+-----------------------------------------------------------------+
| >  | > DMPLEXPARTITIONLABELADJACENCY(lblOl, n)                       |
|  1 |                                                                 |
| 3: |                                                                 |
+----+-----------------------------------------------------------------+

Having established the mapping required to migrate remote overlap
points, we can derive a migration SF similar to the one used in Alg. 5.
As shown in Alg. 8, this allows us to utilize DMPlexMigrate() to
generate the overlapping local sub-meshes, provided the migration SF
also encapsulates the local point renumbering required to maintain
stratification in the DMPlex DAG, meaning that cells are numbered
contiguously, vertices are numbered contiguously, etc. This graph
numbering shift can easily be derived from the SF that encapsulates the
remote point contributions, thus enabling us to express local and remote
components of the overlap migration in a single SF.

**Algorithm 8** Algorithm for migrating overlap points

> 1: **function** DMPLEXDISTRIBUTEOVERLAP(dm, overlap, sf, odm)

2: DMPLEXCREATEOVERLAP(dm, sf, lblOl) *.* Create overlap label

> 3: DMPLEXPARTITIONLABELCREATESF(dm, lblOl, sfOl) *.* Derive migration
> SF 4: DMPLEXSTRATIFYMIGRATIONSF(dm, sfOl, sfMig) *.* Shift point
> numbering

5: DMPLEXMIGRATE(dm, sfMig, dmOl) *.* Distribute overlap

6: DMPLEXDISTRIBUTESF(dm, sfMig, dmOl) *.* Create new SF

# RESULTS

The performance of the distribution algorithms detailed in Alg. 5 and 8
has been evaluated on the UK National Supercomputer ARCHER, a Cray XE30
with 4920 nodes connected via an Aries interconnect [^1]. Each node
consists of two 2.7 GHz, 12-core Intel E5-2697 v2 (Ivy Bridge)
processors with 64GB of memory. The benchmarks consist of distributing a
three dimensional simplicial mesh of the unit cube across increasing
numbers of MPI processes (strong scaling), while measuring execution
time and the total amount of data communicated per processor. The mesh
is generated in memory using TetGen \[Si 2015; Si 2005\] and the
partitioner used is Metis/ParMetis \[Karypis and Kumar 1998; Karypis et
al. 2005\].

The performance of the partitioning and data migration components of the
initial oneto-all mesh distribution, as well as the subsequent
generation of the parallel overlap

> ![](./mesh2/media/image6.png){width="4.89in"
> height="2.0166666666666666in"}

Fig. 4. Performance of initial one-to-all mesh distribution of a 3D unit
cube mesh with approximately 12 million cells. The distribution time is
dominated by the time to send the serial mesh to all processes, and the
overlap determination and communication time scales linearly with the
number of processes.

mapping is detailed in Fig. 4. The presented run-time measurements
indicate that the parallel overlap generation scales linearly with
increasing numbers of processes, whereas the cost of the initial mesh
distribution increases due to the the sequential partitioning cost.

Data communication was measured as the accumulated message volume (sent
and received) per process for each stage using PETSc's performance
logging \[Balay et al. 2014a\]. As expected, the overall communication
volume during distributed overlap generation increases with the number
of processes due to data replication along the shared partition
boundaries. The communicated data volume during the initial
distribution, however, remains constant, indicating that the increasing
run-time cost is due to sequential processing, not communication of the
partitioning. In fact, the number of high-level communication calls,
such as SF-broadcasts and SF-reductions is constant for meshes of all
sizes and numbers of processes. A model of the total data volume
communicated during the initial distribution of a three-dimensional mesh
can be established as follows:

*V~sf\ ~*= 4*B* ∗ *N*

> *Vinversion* = *Vsf* + 2 ∗ 4*B* ∗ *N Vstratify* = *Vsf* + 4*B* ∗ *N*
>
> *Vpartition* = *Vinversion* + *Vstratify*

\(6\)

> *V~cones\ ~*= *N~c\ ~*∗ 4*B* ∗ 4 + *N~f\ ~*∗ 4*B* ∗ 3 + *N~e\ ~*∗ 4*B*
> ∗ 2
>
> *Vorientations* = *Nc* ∗ 4*B* ∗ 4 + *Nf* ∗ 4*B* ∗ 3 + *Ne* ∗ 4*B* ∗ 2
> *Vsection* = 3 ∗ *Vsf* + 2 ∗ 4*B* ∗ *N*
>
> *Vtopology* = *Vcones* + *Vorientations* + *Vsection*
>
> *Vcoordinates* = (3 ∗ 8*B* + 2 ∗ 4*B*) ∗ *Nv*
>
> *Vmarkers* = 3 ∗ *Vsf*

*Vmigration* = *Vtopology* + *Vcoordinates* + *Vmarkers* (7)

where *N~c~*, *N~f~*, *N~e\ ~*and *N~v\ ~*denote the number of cells,
faces, edges and vertices respectively, *N* = *N~c\ ~*+ *N~f\ ~*+
*N~e\ ~*+ *N~v\ ~*and *V~sf\ ~*is the data volume required to initialize
an SF. The unit square mesh used in the benchmarks has *N~c\ ~*=
12*,*582*,*912, *N~f\ ~*= 25*,*264*,*128, *N~e\ ~*= 14*,*827*,*904,
*N~v\ ~*= 2*,*146*,*689, resulting in *V~partition\ ~*≈ 1*.*1*GB* and
*V~migration\ ~*≈ 2*.*8*GB*.

As well as initial mesh distribution the presented API also allows
all-to-all mesh distribution in order to improve load balance among
partitions. Fig. 5 depicts run-time and memory measurements for such a
redistribution process, where an initial bad partitioning based on
random assignment is improved through re-partitioning with ParMETIS.
Similarly to the overlap distribution, the run-time cost demonstrate
good scalability for the partitioning as well as the migration phase,
while the communication volume increases with the number of processes.

> ![](./mesh2/media/image7.png){width="4.89in"
> height="2.0166666666666666in"}

Fig. 5. Performance of all-to-all mesh distribution of simplicial meshes
in 2D and 3D. An initial random partitioning is re-partitioned via
ParMetis and re-distributed to achieve load balancing.

As demonstrated in Fig. 4, the sequential overhead of generating the
base mesh on a single process limits overall scalability of parallel
run-time mesh generation. To overcome this bottleneck, parallel mesh
refinement can be used to create high-resolution meshes in parallel from
an initial coarse mesh. The performance benefits of this approach are
highlighted in Fig. 6, where regular refinement is applied to a unit
cube mesh with varying numbers of edges in each dimension. The
performance measurements show clear improvements for the sequential
components, initial mesh generation and distribution, through the
reduced mesh size, while the parallel refinement operations and
subsequent overlap generation scale linearly. Such an approach is
particularly useful for the generation of mesh hierarchies required for
multigrid preconditioning.

> ![](./mesh2/media/image8.png){width="3.74in"
> height="1.9933333333333334in"}

Fig. 6. Performance of parallel mesh generation via regular three
dimensional refinement.

# CONCLUSIONS

We have developed a concise, powerful API for general parallel mesh
manipulation, based upon the DMPlexMigrate() capability. With just a few
methods, we are able to express mesh distribution from serial to
parallel, parallel redistribution of a mesh, and parallel determination
and communication of arbitrary overlap. Moreover, a user could combine
these facilities to specialize mesh distribution for certain parts of a
calculation or for certain fields or solvers, since they are not
constrained by a monolithic interface. Moreover, the same code applies
to meshes of any dimension, with any cell shape and connectivity. Thus
optimization of these few routines would apply to the universe of meshes
expressible as CW-complexes. In combination with a set of widely used
mesh file format readers this provides a powerful set of tools for
efficient mesh management available to a wide range of applications
through PETSc library interfaces \[Lange et al. 2015\].

In future work, we will apply these building blocks to the problem of
fully parallel mesh construction and adaptivity. We will input a naive
partition of the mesh calculable from common serial mesh formats, and
then rebalance the mesh in parallel. We are developing an interface to
the Pragmatic unstructured parallel mesh refinement package \[Rokos and
Gorman 2013\], which will allow parallel adaptive refinement where we
currently use only regular refinement.

# REFERENCES {#references .unnumbered}

BALAY, S., ABHYANKAR, S., ADAMS, M. F., BROWN, J., BRUNE, P.,
BUSCHELMAN, K., EIJKHOUT, V., GROPP, W. D., KAUSHIK, D., KNEPLEY, M. G.,
MCINNES, L. C., RUPP, K., SMITH, B. F., AND ZHANG, H. 2014a. PETSc users
manual. Tech. Rep. ANL-95/11 - Revision 3.5, Argonne National
Laboratory.

BALAY, S., ABHYANKAR, S., ADAMS, M. F., BROWN, J., BRUNE, P.,
BUSCHELMAN, K., EIJKHOUT, V., GROPP, W. D., KAUSHIK, D., KNEPLEY, M. G.,
MCINNES, L. C., RUPP, K., SMITH, B. F., AND ZHANG, H. 2014b. PETSc Web
page. [http://www.mcs.anl.gov/petsc.](http://www.mcs.anl.gov/petsc)

BIRKHOFF, G. 1967. *Lattice theory*. Vol. 25. American Mathematical
Society.

BROWN, J. 2011. Star forests as a parallel communication model.

D'AZEVEDO, E. AND ET. AL. 2015. Itaps web site.

DEDNER, A., KLÖFKORN, R., NOLTE, M., AND OHLBERGER, M. 2010. A generic
interface for parallel and adaptive discretization schemes: abstraction
principles and the dune-fem module. *Computing 90,* 3-4, 165--196.

DEVINE, K. D., BOMAN, E. G., HEAPHY, R. T., ÇATALYÜREK, U. V., AND
BISSELING, R. H. 2006. Parallel hypergraph partitioning for irregular
problems. *SIAM Parallel Processing for Scientific Computing*.

HATCHER, A. 2002. *Algebraic topology*. Cambridge University Press.

HOEFLER, T., SIEBERT, C., AND LUMSDAINE, A. 2010. Scalable Communication
Protocols for Dynamic Sparse Data Exchange. In *Proceedings of the 2010
ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming
(PPoPP'10)*. ACM, 159--168.

KARYPIS, G. AND KUMAR, V. 1998. A parallel algorithm for multilevel
graph partitioning and sparse matrix ordering. *Journal of Parallel and
Distributed Computing 48*, 71--85.

KARYPIS ET AL., G. 2005. ParMETIS Web page.
[http://www.cs.umn.edu/\~karypis/metis/parmetis.](http://www.cs.umn.edu/~karypis/metis/parmetis)

KNEPLEY, M. G. AND KARPEEV, D. A. 2009. Mesh algorithms for PDE with
Sieve I: Mesh distribution. *Scientific Programming 17,* 3, 215--230.
[http://arxiv.org/abs/0908.4427.](http://arxiv.org/abs/0908.4427)

LANGE, M., KNEPLEY, M. G., AND GORMAN, G. J. 2015. Flexible, scalable
mesh and data management using petsc dmplex.

ROKOS, G. AND GORMAN, G. 2013. Pragmatic--parallel anisotropic adaptive
mesh toolkit. In *Facing the Multicore-Challenge III*. Springer,
143--144.

SI, H. 2005. TetGen: A Quality Tetrahedral Mesh Generator and
Three-Dimensional Delaunay Triangulator.
[http://tetgen.berlios.de.](http://tetgen.berlios.de/)

SI, H. 2015. Tetgen, a delaunay-based quality tetrahedral mesh
generator. *ACM Trans. on Mathematical Software 41,* 2.

TAYLOR, C. AND HOOD, P. 1973. A numerical solution of the navier-stokes
equations using the finite element technique. *Computers & Fluids 1,* 1,
73--100.

WIKIPEDIA. 2015a. Cw complex.
[http://en.wikipedia.org/wiki/CW_complex.](http://en.wikipedia.org/wiki/CW_complex)

WIKIPEDIA. 2015b. Hasse diagram.
[http://en.wikipedia.org/wiki/Hasse_diagram.](http://en.wikipedia.org/wiki/Hasse_diagram)

[^1]: <http://www.archer.ac.uk/>

## code

1. `src/domain/mesh_entity.rs`:

```rust
// src/domain/mesh_entity.rs

/// Represents an entity in a mesh, such as a vertex, edge, face, or cell, 
/// using a unique identifier for each entity type.  
/// 
/// The `MeshEntity` enum defines four types of mesh entities:
/// - Vertex: Represents a point in the mesh.
/// - Edge: Represents a connection between two vertices.
/// - Face: Represents a polygonal area bounded by edges.
/// - Cell: Represents a volumetric region of the mesh.
///
/// Example usage:
///
///    let vertex = MeshEntity::Vertex(1);  
///    let edge = MeshEntity::Edge(2);  
///    assert_eq!(vertex.id(), 1);  
///    assert_eq!(vertex.entity_type(), "Vertex");  
///    assert_eq!(edge.id(), 2);  
/// 
#[derive(Debug, Hash, Eq, PartialEq, PartialOrd, Clone, Copy)]
pub enum MeshEntity {
    Vertex(usize),  // Vertex id 
    Edge(usize),    // Edge id 
    Face(usize),    // Face id 
    Cell(usize),    // Cell id 
}

impl MeshEntity {
    /// Returns the unique identifier associated with the `MeshEntity`.  
    ///
    /// This function matches the enum variant and returns the id for that 
    /// particular entity (e.g., for a `Vertex`, it will return the vertex id).  
    ///
    /// Example usage:
    /// 
    ///    let vertex = MeshEntity::Vertex(3);  
    ///    assert_eq!(vertex.id(), 3);  
    ///
    pub fn id(&self) -> usize {
        match *self {
            MeshEntity::Vertex(id) => id,
            MeshEntity::Edge(id) => id,
            MeshEntity::Face(id) => id,
            MeshEntity::Cell(id) => id,
        }
    }

    /// Returns the type of the `MeshEntity` as a string, indicating whether  
    /// the entity is a Vertex, Edge, Face, or Cell.  
    ///
    /// Example usage:
    /// 
    ///    let face = MeshEntity::Face(1);  
    ///    assert_eq!(face.entity_type(), "Face");  
    ///
    pub fn entity_type(&self) -> &str {
        match *self {
            MeshEntity::Vertex(_) => "Vertex",
            MeshEntity::Edge(_) => "Edge",
            MeshEntity::Face(_) => "Face",
            MeshEntity::Cell(_) => "Cell",
        }
    }
}

/// A struct representing a directed relationship between two mesh entities,  
/// known as an `Arrow`. It holds the "from" and "to" entities, representing  
/// a connection from one entity to another.  
///
/// Example usage:
/// 
///    let from = MeshEntity::Vertex(1);  
///    let to = MeshEntity::Edge(2);  
///    let arrow = Arrow::new(from, to);  
///    let (start, end) = arrow.get_relation();  
///    assert_eq!(*start, MeshEntity::Vertex(1));  
///    assert_eq!(*end, MeshEntity::Edge(2));  
/// 
pub struct Arrow {
    pub from: MeshEntity,  // The starting entity of the relation 
    pub to: MeshEntity,    // The ending entity of the relation 
}

impl Arrow {
    /// Creates a new `Arrow` between two mesh entities.  
    ///
    /// Example usage:
    /// 
    ///    let from = MeshEntity::Cell(1);  
    ///    let to = MeshEntity::Face(3);  
    ///    let arrow = Arrow::new(from, to);  
    ///
    pub fn new(from: MeshEntity, to: MeshEntity) -> Self {
        Arrow { from, to }
    }

    /// Converts a generic entity type that implements `Into<MeshEntity>` into  
    /// a `MeshEntity`.  
    ///
    /// Example usage:
    /// 
    ///    let vertex = MeshEntity::Vertex(5);  
    ///    let entity = Arrow::add_entity(vertex);  
    ///    assert_eq!(entity.id(), 5);  
    ///
    pub fn add_entity<T: Into<MeshEntity>>(entity: T) -> MeshEntity {
        entity.into()
    }

    /// Returns a tuple reference of the "from" and "to" entities of the `Arrow`.  
    ///
    /// Example usage:
    /// 
    ///    let from = MeshEntity::Edge(1);  
    ///    let to = MeshEntity::Face(2);  
    ///    let arrow = Arrow::new(from, to);  
    ///    let (start, end) = arrow.get_relation();  
    ///    assert_eq!(*start, MeshEntity::Edge(1));  
    ///    assert_eq!(*end, MeshEntity::Face(2));  
    ///
    pub fn get_relation(&self) -> (&MeshEntity, &MeshEntity) {
        (&self.from, &self.to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test that verifies the id and type of a `MeshEntity` are correctly returned.  
    fn test_entity_id_and_type() {
        let vertex = MeshEntity::Vertex(1);
        assert_eq!(vertex.id(), 1);
        assert_eq!(vertex.entity_type(), "Vertex");
    }

    #[test]
    /// Test that verifies the creation of an `Arrow` and the correctness of  
    /// the `get_relation` function.  
    fn test_arrow_creation_and_relation() {
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(2);
        let arrow = Arrow::new(vertex, edge);
        let (from, to) = arrow.get_relation();
        assert_eq!(*from, MeshEntity::Vertex(1));
        assert_eq!(*to, MeshEntity::Edge(2));
    }

    #[test]
    /// Test that verifies the addition of an entity using the `add_entity` function.  
    fn test_add_entity() {
        let vertex = MeshEntity::Vertex(5);
        let added_entity = Arrow::add_entity(vertex);

        assert_eq!(added_entity.id(), 5);
        assert_eq!(added_entity.entity_type(), "Vertex");
    }
}
```

2. `src/domain/sieve.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

/// A `Sieve` struct that manages the relationships (arrows) between `MeshEntity`  
/// elements, organized in an adjacency map.
///
/// The adjacency map tracks directed relations between entities in the mesh.  
/// It supports operations such as adding relationships, querying direct  
/// relations (cones), and computing closure and star sets for entities.
#[derive(Clone, Debug)]
pub struct Sieve {
    /// A thread-safe adjacency map where each key is a `MeshEntity`,  
    /// and the value is a set of `MeshEntity` objects related to the key.  
    pub adjacency: DashMap<MeshEntity, DashMap<MeshEntity, ()>>,
}

impl Sieve {
    /// Creates a new empty `Sieve` instance with an empty adjacency map.
    pub fn new() -> Self {
        Sieve {
            adjacency: DashMap::new(),
        }
    }

    /// Adds a directed relationship (arrow) between two `MeshEntity` elements.  
    /// The relationship is stored in the adjacency map from the `from` entity  
    /// to the `to` entity.
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.adjacency
            .entry(from)
            .or_insert_with(DashMap::new)
            .insert(to, ());
    }

    /// Retrieves all entities directly related to the given entity (`point`).  
    /// This operation is referred to as retrieving the cone of the entity.  
    /// Returns `None` if there are no related entities.
    pub fn cone(&self, point: &MeshEntity) -> Option<Vec<MeshEntity>> {
        self.adjacency.get(point).map(|cone| {
            cone.iter().map(|entry| entry.key().clone()).collect()
        })
    }

    /// Computes the closure of a given `MeshEntity`.  
    /// The closure includes the entity itself and all entities it covers (cones) recursively.
    pub fn closure(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        let stack = DashMap::new();
        stack.insert(point.clone(), ());

        while !stack.is_empty() {
            let keys: Vec<MeshEntity> = stack.iter().map(|entry| entry.key().clone()).collect();
            for p in keys {
                if result.insert(p.clone(), ()).is_none() {
                    if let Some(cones) = self.cone(&p) {
                        for q in cones {
                            stack.insert(q, ());
                        }
                    }
                }
                stack.remove(&p);
            }
        }
        result
    }

    /// Computes the star of a given `MeshEntity`.  
    /// The star includes the entity itself and all entities that directly cover it (supports).
    pub fn star(&self, point: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let result = DashMap::new();
        result.insert(point.clone(), ());
        let supports = self.support(point);
        for support in supports {
            result.insert(support, ());
        }
        result
    }

    /// Retrieves all entities that support the given entity (`point`).  
    /// These are the entities that have an arrow pointing to `point`.
    pub fn support(&self, point: &MeshEntity) -> Vec<MeshEntity> {
        let mut supports = Vec::new();
        self.adjacency.iter().for_each(|entry| {
            let from = entry.key();
            if entry.value().contains_key(point) {
                supports.push(from.clone());
            }
        });
        supports
    }

    /// Computes the meet operation for two entities, `p` and `q`.  
    /// This is the intersection of their closures.
    pub fn meet(&self, p: &MeshEntity, q: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let closure_p = self.closure(p);
        let closure_q = self.closure(q);
        let result = DashMap::new();

        closure_p.iter().for_each(|entry| {
            let key = entry.key();
            if closure_q.contains_key(key) {
                result.insert(key.clone(), ());
            }
        });

        result
    }

    /// Computes the join operation for two entities, `p` and `q`.  
    /// This is the union of their stars.
    pub fn join(&self, p: &MeshEntity, q: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let star_p = self.star(p);
        let star_q = self.star(q);
        let result = DashMap::new();

        star_p.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });
        star_q.iter().for_each(|entry| {
            result.insert(entry.key().clone(), ());
        });

        result
    }

    /// Applies a given function in parallel to all adjacency map entries.  
    /// This function is executed concurrently over each entity and its  
    /// corresponding set of related entities.
    pub fn par_for_each_adjacent<F>(&self, func: F)
    where
        F: Fn((&MeshEntity, Vec<MeshEntity>)) + Sync + Send,
    {
        // Collect entries from DashMap to avoid borrow conflicts
        let entries: Vec<_> = self.adjacency.iter().map(|entry| {
            let key = entry.key().clone();
            let values: Vec<MeshEntity> = entry.value().iter().map(|e| e.key().clone()).collect();
            (key, values)
        }).collect();

        // Execute in parallel over collected entries
        entries.par_iter().for_each(|entry| {
            func((&entry.0, entry.1.clone()));
        });
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    /// Test that verifies adding an arrow between two entities and querying  
    /// the cone of an entity works as expected.
    fn test_add_arrow_and_cone() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        sieve.add_arrow(vertex, edge);
        let cone_result = sieve.cone(&vertex).unwrap();
        assert!(cone_result.contains(&edge));
    }

    #[test]
    /// Test that verifies the closure of a vertex correctly includes  
    /// all transitive relationships and the entity itself.
    fn test_closure() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);
        sieve.add_arrow(vertex, edge);
        sieve.add_arrow(edge, face);
        let closure_result = sieve.closure(&vertex);
        assert!(closure_result.contains_key(&vertex));
        assert!(closure_result.contains_key(&edge));
        assert!(closure_result.contains_key(&face));
        assert_eq!(closure_result.len(), 3);
    }

    #[test]
    /// Test that verifies the support of an entity includes the  
    /// correct supporting entities.
    fn test_support() {
        let sieve = Sieve::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex, edge);
        let support_result = sieve.support(&edge);

        assert!(support_result.contains(&vertex));
        assert_eq!(support_result.len(), 1);
    }

    #[test]
    /// Test that verifies the star of an entity includes both the entity itself and  
    /// its immediate supports.
    fn test_star() {
        let sieve = Sieve::new();
        let edge = MeshEntity::Edge(1);
        let face = MeshEntity::Face(1);

        sieve.add_arrow(edge, face);

        let star_result = sieve.star(&face);

        assert!(star_result.contains_key(&face));
        assert!(star_result.contains_key(&edge));
        assert_eq!(star_result.len(), 2);
    }

    #[test]
    /// Test that verifies the meet operation between two entities returns  
    /// the correct intersection of their closures.
    fn test_meet() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let edge = MeshEntity::Edge(1);

        sieve.add_arrow(vertex1, edge);
        sieve.add_arrow(vertex2, edge);

        let meet_result = sieve.meet(&vertex1, &vertex2);

        assert!(meet_result.contains_key(&edge));
        assert_eq!(meet_result.len(), 1);
    }

    #[test]
    /// Test that verifies the join operation between two entities returns  
    /// the correct union of their stars.
    fn test_join() {
        let sieve = Sieve::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        let join_result = sieve.join(&vertex1, &vertex2);

        assert!(join_result.contains_key(&vertex1), "Join result should contain vertex1");
        assert!(join_result.contains_key(&vertex2), "Join result should contain vertex2");
        assert_eq!(join_result.len(), 2);
    }
}

```

3. `src/domain/mesh/mod.rs`

```rust
pub mod entities;
pub mod geometry;
pub mod reordering;
pub mod boundary;
pub mod hierarchical;

use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{Arc, RwLock};
use crossbeam::channel::{Sender, Receiver};

// Delegate methods to corresponding modules

/// Represents the mesh structure, which is composed of a sieve for entity management,  
/// a set of mesh entities, vertex coordinates, and channels for boundary data.  
/// 
/// The `Mesh` struct is the central component for managing mesh entities and  
/// their relationships. It stores entities such as vertices, edges, faces,  
/// and cells, along with their geometric data and boundary-related information.  
/// 
/// Example usage:
/// 
///    let mesh = Mesh::new();  
///    let entity = MeshEntity::Vertex(1);  
///    mesh.entities.write().unwrap().insert(entity);  
/// 
#[derive(Clone)]
pub struct Mesh {
    /// The sieve structure used for organizing the mesh entities' relationships.  
    pub sieve: Arc<Sieve>,  
    /// A thread-safe, read-write lock for managing mesh entities.  
    /// This set contains all `MeshEntity` objects in the mesh.  
    pub entities: Arc<RwLock<FxHashSet<MeshEntity>>>,  
    /// A map from vertex indices to their 3D coordinates.  
    pub vertex_coordinates: FxHashMap<usize, [f64; 3]>,  
    /// An optional channel sender for transmitting boundary data related to mesh entities.  
    pub boundary_data_sender: Option<Sender<FxHashMap<MeshEntity, [f64; 3]>>>,  
    /// An optional channel receiver for receiving boundary data related to mesh entities.  
    pub boundary_data_receiver: Option<Receiver<FxHashMap<MeshEntity, [f64; 3]>>>,  
}

impl Mesh {
    /// Creates a new instance of the `Mesh` struct with initialized components.  
    /// 
    /// This method sets up the sieve, entity set, vertex coordinate map,  
    /// and a channel for boundary data communication between mesh components.  
    ///
    /// The `Sender` and `Receiver` are unbounded channels used to pass boundary  
    /// data between mesh modules asynchronously.
    /// 
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    assert!(mesh.entities.read().unwrap().is_empty());  
    /// 
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam::channel::unbounded();
        Mesh {
            sieve: Arc::new(Sieve::new()),
            entities: Arc::new(RwLock::new(FxHashSet::default())),
            vertex_coordinates: FxHashMap::default(),
            boundary_data_sender: Some(sender),
            boundary_data_receiver: Some(receiver),
        }
    }
}

#[cfg(test)]
pub mod tests;
```

4. `src/domain/mesh/entities.rs`

```rust

use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

impl Mesh {
    /// Adds a new `MeshEntity` to the mesh.  
    /// The entity will be inserted into the thread-safe `entities` set.  
    /// 
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    mesh.add_entity(vertex);  
    /// 
    pub fn add_entity(&self, entity: MeshEntity) {
        self.entities.write().unwrap().insert(entity);
    }

    /// Establishes a relationship (arrow) between two mesh entities.  
    /// This creates an arrow from the `from` entity to the `to` entity  
    /// in the sieve structure.  
    ///
    /// Example usage:
    /// 
    ///    let mut mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(2);  
    ///    mesh.add_relationship(vertex, edge);  
    /// 
    pub fn add_relationship(&mut self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Adds an arrow from one mesh entity to another in the sieve structure.  
    /// This method is a simple delegate to the `Sieve`'s `add_arrow` method.
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    let edge = MeshEntity::Edge(2);  
    ///    mesh.add_arrow(vertex, edge);  
    /// 
    pub fn add_arrow(&self, from: MeshEntity, to: MeshEntity) {
        self.sieve.add_arrow(from, to);
    }

    /// Sets the 3D coordinates for a vertex and adds the vertex entity  
    /// to the mesh if it's not already present.  
    /// 
    /// This method inserts the vertex's coordinates into the  
    /// `vertex_coordinates` map and adds the vertex to the `entities` set.
    ///
    /// Example usage:
    /// 
    ///    let mut mesh = Mesh::new();  
    ///    mesh.set_vertex_coordinates(1, [1.0, 2.0, 3.0]);  
    ///    assert_eq!(mesh.get_vertex_coordinates(1), Some([1.0, 2.0, 3.0]));  
    ///
    pub fn set_vertex_coordinates(&mut self, vertex_id: usize, coords: [f64; 3]) {
        self.vertex_coordinates.insert(vertex_id, coords);
        self.add_entity(MeshEntity::Vertex(vertex_id));
    }

    /// Retrieves the 3D coordinates of a vertex by its identifier.  
    ///
    /// Returns `None` if the vertex does not exist in the `vertex_coordinates` map.
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let coords = mesh.get_vertex_coordinates(1);  
    ///    assert!(coords.is_none());  
    ///
    pub fn get_vertex_coordinates(&self, vertex_id: usize) -> Option<[f64; 3]> {
        self.vertex_coordinates.get(&vertex_id).cloned()
    }

    /// Counts the number of entities of a specified type (e.g., Vertex, Edge, Face, Cell)  
    /// within the mesh.  
    ///
    /// Example usage:
    /// 
    ///    let mesh = Mesh::new();  
    ///    let count = mesh.count_entities(&MeshEntity::Vertex(1));  
    ///    assert_eq!(count, 0);  
    ///
    pub fn count_entities(&self, entity_type: &MeshEntity) -> usize {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| match (e, entity_type) {
                (MeshEntity::Vertex(_), MeshEntity::Vertex(_)) => true,
                (MeshEntity::Cell(_), MeshEntity::Cell(_)) => true,
                (MeshEntity::Edge(_), MeshEntity::Edge(_)) => true,
                (MeshEntity::Face(_), MeshEntity::Face(_)) => true,
                _ => false,
            })
            .count()
    }

    /// Applies a given function to each entity in the mesh in parallel.  
    ///
    /// The function `func` is applied to all mesh entities concurrently  
    /// using Rayon’s parallel iterator.
    ///
    /// Example usage:
    /// 
    ///    mesh.par_for_each_entity(|entity| {  
    ///        println!("{:?}", entity);  
    ///    });  
    ///
    pub fn par_for_each_entity<F>(&self, func: F)
    where
        F: Fn(&MeshEntity) + Sync + Send,
    {
        let entities = self.entities.read().unwrap();
        entities.par_iter().for_each(func);
    }

    /// Retrieves all the `Cell` entities from the mesh.  
    ///
    /// This method returns a `Vec<MeshEntity>` containing all entities  
    /// classified as cells.
    ///
    /// Example usage:
    /// 
    ///    let cells = mesh.get_cells();  
    ///    assert!(cells.is_empty());  
    ///
    pub fn get_cells(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Cell(_)))
            .cloned()
            .collect()
    }

    /// Retrieves all the `Face` entities from the mesh.  
    ///
    /// This method returns a `Vec<MeshEntity>` containing all entities  
    /// classified as faces.
    ///
    /// Example usage:
    /// 
    ///    let faces = mesh.get_faces();  
    ///    assert!(faces.is_empty());  
    ///
    pub fn get_faces(&self) -> Vec<MeshEntity> {
        let entities = self.entities.read().unwrap();
        entities.iter()
            .filter(|e| matches!(e, MeshEntity::Face(_)))
            .cloned()
            .collect()
    }

    /// Computes properties for each entity in the mesh in parallel,  
    /// returning a map of `MeshEntity` to the computed property.  
    ///
    /// The `compute_fn` is a user-provided function that takes a reference  
    /// to a `MeshEntity` and returns a computed value of type `PropertyType`.  
    ///
    /// Example usage:
    /// 
    ///    let properties = mesh.compute_properties(|entity| {  
    ///        entity.id()  
    ///    });  
    ///
    pub fn compute_properties<F, PropertyType>(&self, compute_fn: F) -> FxHashMap<MeshEntity, PropertyType>
    where
        F: Fn(&MeshEntity) -> PropertyType + Sync + Send,
        PropertyType: Send,
    {
        let entities = self.entities.read().unwrap();
        entities
            .par_iter()
            .map(|entity| (*entity, compute_fn(entity)))
            .collect()
    }
}

```

5. `src/domain/mesh/geometry.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use crate::geometry::{Geometry, CellShape, FaceShape};
use dashmap::DashMap;

impl Mesh {
    /// Retrieves all the faces of a given cell.  
    ///
    /// This method uses the `cone` function of the sieve to obtain all the faces  
    /// connected to the given cell.  
    ///
    /// Returns a set of `MeshEntity` representing the faces of the cell, or  
    /// `None` if the cell has no connected faces.  
    ///
    pub fn get_faces_of_cell(&self, cell: &MeshEntity) -> Option<DashMap<MeshEntity, ()>> {
        self.sieve.cone(cell).map(|set| {
            let faces = DashMap::new();
            set.into_iter().for_each(|face| { faces.insert(face, ()); });
            faces
        })
    }

    /// Retrieves all the cells that share the given face.  
    ///
    /// This method uses the `support` function of the sieve to obtain all the cells  
    /// that are connected to the given face.  
    ///
    /// Returns a set of `MeshEntity` representing the neighboring cells.  
    ///
    pub fn get_cells_sharing_face(&self, face: &MeshEntity) -> DashMap<MeshEntity, ()> {
        let cells = DashMap::new();
        self.sieve.support(face).into_iter().for_each(|cell| { cells.insert(cell, ()); });
        cells
    }

    /// Computes the Euclidean distance between two cells based on their centroids.  
    ///
    /// This method calculates the centroids of both cells and then uses the `Geometry`  
    /// module to compute the distance between these centroids.  
    ///
    pub fn get_distance_between_cells(&self, cell_i: &MeshEntity, cell_j: &MeshEntity) -> f64 {
        let centroid_i = self.get_cell_centroid(cell_i);
        let centroid_j = self.get_cell_centroid(cell_j);
        Geometry::compute_distance(&centroid_i, &centroid_j)
    }

    /// Computes the area of a face based on its geometric shape and vertices.  
    ///
    /// This method determines the face shape (triangle or quadrilateral) and  
    /// uses the `Geometry` module to compute the area.  
    ///
    pub fn get_face_area(&self, face: &MeshEntity) -> f64 {
        let face_vertices = self.get_face_vertices(face);
        let face_shape = match face_vertices.len() {
            3 => FaceShape::Triangle,
            4 => FaceShape::Quadrilateral,
            _ => panic!("Unsupported face shape with {} vertices", face_vertices.len()),
        };

        let mut geometry = Geometry::new();
        let face_id = face.id();
        geometry.compute_face_area(face_id, face_shape, &face_vertices)
    }

    /// Computes the centroid of a cell based on its vertices.  
    ///
    /// This method determines the cell shape and uses the `Geometry` module to compute the centroid.  
    ///
    pub fn get_cell_centroid(&self, cell: &MeshEntity) -> [f64; 3] {
        let cell_vertices = self.get_cell_vertices(cell);
        let _cell_shape = match cell_vertices.len() {
            4 => CellShape::Tetrahedron,
            5 => CellShape::Pyramid,
            6 => CellShape::Prism,
            8 => CellShape::Hexahedron,
            _ => panic!("Unsupported cell shape with {} vertices", cell_vertices.len()),
        };

        let mut geometry = Geometry::new();
        geometry.compute_cell_centroid(self, cell)
    }

    /// Retrieves all vertices connected to the given vertex by shared cells.  
    ///
    /// This method uses the `support` function of the sieve to find cells that  
    /// contain the given vertex and then retrieves all other vertices in those cells.  
    ///
    pub fn get_neighboring_vertices(&self, vertex: &MeshEntity) -> Vec<MeshEntity> {
        let neighbors = DashMap::new();
        let connected_cells = self.sieve.support(vertex);

        connected_cells.into_iter().for_each(|cell| {
            if let Some(cell_vertices) = self.sieve.cone(&cell).as_ref() {
                for v in cell_vertices {
                    if v != vertex && matches!(v, MeshEntity::Vertex(_)) {
                        neighbors.insert(v.clone(), ());
                    }
                }
            } else {
                panic!("Cell {:?} has no connected vertices", cell);
            }
        });
        neighbors.into_iter().map(|(vertex, _)| vertex).collect()
    }

    /// Returns an iterator over the IDs of all vertices in the mesh.  
    ///
    pub fn iter_vertices(&self) -> impl Iterator<Item = &usize> {
        self.vertex_coordinates.keys()
    }

    /// Determines the shape of a cell based on the number of vertices it has.  
    ///
    pub fn get_cell_shape(&self, cell: &MeshEntity) -> Result<CellShape, String> {
        let cell_vertices = self.get_cell_vertices(cell);
        match cell_vertices.len() {
            4 => Ok(CellShape::Tetrahedron),
            5 => Ok(CellShape::Pyramid),
            6 => Ok(CellShape::Prism),
            8 => Ok(CellShape::Hexahedron),
            _ => Err(format!(
                "Unsupported cell shape with {} vertices. Expected 4, 5, 6, or 8 vertices.",
                cell_vertices.len()
            )),
        }
    }

    /// Retrieves the vertices of a cell and their coordinates.  
    ///
    pub fn get_cell_vertices(&self, cell: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
    
        if let Some(connected_entities) = self.sieve.cone(cell) {
            for entity in connected_entities {
                if let MeshEntity::Vertex(vertex_id) = entity {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertices.push(coords);
                    } else {
                        panic!("Coordinates for vertex {} not found", vertex_id);
                    }
                }
            }
        }

        vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vertices.dedup();
        vertices
    }

    /// Retrieves the vertices of a face and their coordinates.  
    ///
    pub fn get_face_vertices(&self, face: &MeshEntity) -> Vec<[f64; 3]> {
        let mut vertices = Vec::new();
        if let Some(connected_vertices) = self.sieve.cone(face) {
            for vertex in connected_vertices {
                if let MeshEntity::Vertex(vertex_id) = vertex {
                    if let Some(coords) = self.get_vertex_coordinates(vertex_id) {
                        vertices.push(coords);
                    }
                }
            }
        }
        vertices
    }
}
```

6. `src/domain/mesh/boundary.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::FxHashMap;
use crossbeam::channel::{Sender, Receiver};

impl Mesh {
    /// Synchronizes the boundary data by first sending the local boundary data  
    /// and then receiving any updated boundary data from other sources.  
    ///
    /// This function ensures that boundary data, such as vertex coordinates,  
    /// is consistent across all mesh partitions.  
    ///
    /// Example usage:
    /// 
    ///    mesh.sync_boundary_data();  
    ///
    pub fn sync_boundary_data(&mut self) {
        self.send_boundary_data();
        self.receive_boundary_data();
    }

    /// Sets the communication channels for boundary data transmission.  
    ///
    /// The sender channel is used to transmit the local boundary data, and  
    /// the receiver channel is used to receive boundary data from other  
    /// partitions or sources.  
    ///
    /// Example usage:
    /// 
    ///    mesh.set_boundary_channels(sender, receiver);  
    ///
    pub fn set_boundary_channels(
        &mut self,
        sender: Sender<FxHashMap<MeshEntity, [f64; 3]>>,
        receiver: Receiver<FxHashMap<MeshEntity, [f64; 3]>>,
    ) {
        self.boundary_data_sender = Some(sender);
        self.boundary_data_receiver = Some(receiver);
    }

    /// Receives boundary data from the communication channel and updates the mesh.  
    ///
    /// This method listens for incoming boundary data (such as vertex coordinates)  
    /// from the receiver channel and updates the local mesh entities and coordinates.  
    ///
    /// Example usage:
    /// 
    ///    mesh.receive_boundary_data();  
    ///
    pub fn receive_boundary_data(&mut self) {
        if let Some(ref receiver) = self.boundary_data_receiver {
            if let Ok(boundary_data) = receiver.recv() {
                let mut entities = self.entities.write().unwrap();
                for (entity, coords) in boundary_data {
                    // Update vertex coordinates if the entity is a vertex.
                    if let MeshEntity::Vertex(id) = entity {
                        self.vertex_coordinates.insert(id, coords);
                    }
                    entities.insert(entity);
                }
            }
        }
    }

    /// Sends the local boundary data (such as vertex coordinates) through  
    /// the communication channel to other partitions or sources.  
    ///
    /// This method collects the vertex coordinates for all mesh entities  
    /// and sends them using the sender channel.  
    ///
    /// Example usage:
    /// 
    ///    mesh.send_boundary_data();  
    ///
    pub fn send_boundary_data(&self) {
        if let Some(ref sender) = self.boundary_data_sender {
            let mut boundary_data = FxHashMap::default();
            let entities = self.entities.read().unwrap();
            for entity in entities.iter() {
                if let MeshEntity::Vertex(id) = entity {
                    if let Some(coords) = self.vertex_coordinates.get(id) {
                        boundary_data.insert(*entity, *coords);
                    }
                }
            }

            // Send the boundary data through the sender channel.
            if let Err(e) = sender.send(boundary_data) {
                eprintln!("Failed to send boundary data: {:?}", e);
            }
        }
    }
}
```

There is more source code remaining to be provided. Please acknowledge you have read and analyzed the material provided so far and have stored this knowledge in your memory.

`src/domain/mesh/reordering.rs`

```rust
use super::Mesh;
use crate::domain::mesh_entity::MeshEntity;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use rayon::prelude::*;

/// Reorders mesh entities using the Cuthill-McKee algorithm.  
/// This algorithm improves memory locality by reducing the bandwidth of sparse matrices,  
/// which is beneficial for solver optimizations.  
///
/// The algorithm starts from the node with the smallest degree and visits its neighbors  
/// in increasing order of their degree.
///
/// Example usage:
/// 
///    let ordered_entities = cuthill_mckee(&entities, &adjacency);  
///
pub fn cuthill_mckee(
    entities: &[MeshEntity], 
    adjacency: &FxHashMap<MeshEntity, Vec<MeshEntity>>
) -> Vec<MeshEntity> {
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    let mut ordered = Vec::new();

    // Find the starting entity (node) with the smallest degree.
    if let Some((start, _)) = entities.iter()
        .map(|entity| (entity, adjacency.get(entity).map_or(0, |neighbors| neighbors.len())))
        .min_by_key(|&(_, degree)| degree)
    {
        queue.push_back(*start);
        visited.insert(*start);
    }

    // Perform the Cuthill-McKee reordering.
    while let Some(entity) = queue.pop_front() {
        ordered.push(entity);
        if let Some(neighbors) = adjacency.get(&entity) {
            let mut sorted_neighbors: Vec<_> = neighbors.iter()
                .filter(|&&n| !visited.contains(&n))
                .cloned()
                .collect();
            sorted_neighbors.sort_by_key(|n| adjacency.get(n).map_or(0, |neighbors| neighbors.len()));
            for neighbor in sorted_neighbors {
                queue.push_back(neighbor);
                visited.insert(neighbor);
            }
        }
    }

    ordered
}

impl Mesh {
    /// Applies a reordering to the mesh entities based on the given new order.  
    ///
    /// This method can be used to reorder entities or update a sparse matrix  
    /// structure based on the new ordering.
    ///
    /// Example usage:
    /// 
    ///    mesh.apply_reordering(&new_order);  
    ///
    pub fn apply_reordering(&mut self, _new_order: &[usize]) {
        // Implement the application of reordering to mesh entities or sparse matrix structure.
    }

    /// Computes the reverse Cuthill-McKee (RCM) ordering starting from a given node.  
    ///
    /// This method performs the RCM algorithm to minimize the bandwidth of sparse matrices  
    /// by reordering mesh entities in reverse order of their Cuthill-McKee ordering.  
    ///
    /// Example usage:
    /// 
    ///    let rcm_order = mesh.rcm_ordering(start_node);  
    ///
    pub fn rcm_ordering(&self, start_node: MeshEntity) -> Vec<MeshEntity> {
        let mut visited = FxHashSet::default();
        let mut queue = VecDeque::new();
        let mut ordering = Vec::new();

        queue.push_back(start_node);
        visited.insert(start_node);

        // Perform breadth-first traversal and order nodes by degree.
        while let Some(node) = queue.pop_front() {
            ordering.push(node);
            if let Some(neighbors) = self.sieve.cone(&node) {
                let mut sorted_neighbors: Vec<_> = neighbors
                    .into_iter()
                    .filter(|n| !visited.contains(n))
                    .collect();
                sorted_neighbors.sort_by_key(|n| self.sieve.cone(n).map_or(0, |set| set.len()));
                for neighbor in sorted_neighbors {
                    queue.push_back(neighbor);
                    visited.insert(neighbor);
                }
            }
        }

        // Reverse the ordering to get the RCM order.
        ordering.reverse();
        ordering
    }

    /// Reorders elements in the mesh using Morton order (Z-order curve) for better memory locality.  
    ///
    /// This method applies the Morton order to the given set of 2D elements (with x and y coordinates).  
    /// Morton ordering is a space-filling curve that helps improve memory access patterns  
    /// in 2D meshes or grids.
    ///
    /// Example usage:
    /// 
    ///    mesh.reorder_by_morton_order(&mut elements);  
    ///
    pub fn reorder_by_morton_order(&mut self, elements: &mut [(u32, u32)]) {
        elements.par_sort_by_key(|&(x, y)| Self::morton_order_2d(x, y));
    }

    /// Computes the Morton order (Z-order curve) for a 2D point with coordinates (x, y).  
    ///
    /// This function interleaves the bits of the x and y coordinates to generate  
    /// a single value that represents the Morton order.  
    ///
    /// Example usage:
    /// 
    ///    let morton_order = Mesh::morton_order_2d(10, 20);  
    ///
    pub fn morton_order_2d(x: u32, y: u32) -> u64 {
        // Helper function to interleave the bits of a 32-bit integer.
        fn part1by1(n: u32) -> u64 {
            let mut n = n as u64;
            n = (n | (n << 16)) & 0x0000_0000_ffff_0000;
            n = (n | (n << 8)) & 0x0000_ff00_00ff_0000;
            n = (n | (n << 4)) & 0x00f0_00f0_00f0_00f0;
            n = (n | (n << 2)) & 0x0c30_0c30_0c30_0c30;
            n = (n | (n << 1)) & 0x2222_2222_2222_2222;
            n
        }

        // Interleave the bits of x and y to compute the Morton order.
        part1by1(x) | (part1by1(y) << 1)
    }
}
```

`src/domain/mesh/hierarchical.rs`

```rust
use std::boxed::Box;

/// Represents a hierarchical mesh node, which can either be a leaf (non-refined)  
/// or a branch (refined into smaller child elements).  
/// 
/// In 2D, each branch contains 4 child elements (quadtree), while in 3D, each branch  
/// would contain 8 child elements (octree).
///
/// Example usage:
/// 
///    let mut node = MeshNode::Leaf(10);  
///    node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);  
///    assert!(matches!(node, MeshNode::Branch { .. }));  
/// 
#[derive(Debug, PartialEq)]
pub enum MeshNode<T> {
    /// A leaf node representing an unrefined element containing data of type `T`.  
    Leaf(T),
    
    /// A branch node representing a refined element with child elements.  
    /// The branch contains its own data and an array of 4 child nodes (for 2D).  
    Branch {
        data: T,
        children: Box<[MeshNode<T>; 4]>,  // 2D quadtree; change to `[MeshNode<T>; 8]` for 3D.
    },
}

impl<T: Clone> MeshNode<T> {
    /// Refines a leaf node into a branch with initialized child nodes.  
    ///
    /// The `init_child_data` function is used to generate the data for each child  
    /// element based on the parent node's data.  
    ///
    /// Example usage:
    /// 
    ///    let mut node = MeshNode::Leaf(10);  
    ///    node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);  
    ///
    pub fn refine<F>(&mut self, init_child_data: F)
    where
        F: Fn(&T) -> [T; 4],  // Function to generate child data from the parent.
    {
        if let MeshNode::Leaf(data) = self {
            let children = init_child_data(data);
            *self = MeshNode::Branch {
                data: data.clone(),
                children: Box::new([
                    MeshNode::Leaf(children[0].clone()),
                    MeshNode::Leaf(children[1].clone()),
                    MeshNode::Leaf(children[2].clone()),
                    MeshNode::Leaf(children[3].clone()),
                ]),
            };
        }
    }

    /// Coarsens a branch back into a leaf node by collapsing its child elements.  
    ///
    /// This method turns a branch back into a leaf node, retaining the data of the  
    /// parent node but removing its child elements.  
    ///
    /// Example usage:
    /// 
    ///    node.coarsen();  
    ///
    pub fn coarsen(&mut self) {
        if let MeshNode::Branch { data, .. } = self {
            *self = MeshNode::Leaf(data.clone());
        }
    }

    /// Applies constraints at hanging nodes to ensure continuity between the parent  
    /// and its child elements.  
    ///
    /// This function adjusts the degrees of freedom (DOFs) at the parent node by  
    /// averaging the DOFs from its child elements.  
    ///
    /// Example usage:
    /// 
    ///    node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);  
    ///
    pub fn apply_hanging_node_constraints(&self, parent_dofs: &mut [f64], child_dofs: &mut [[f64; 4]; 4]) {
        if let MeshNode::Branch { .. } = self {
            for i in 0..parent_dofs.len() {
                parent_dofs[i] = child_dofs.iter().map(|d| d[i]).sum::<f64>() / 4.0;
            }
        }
    }

    /// Returns an iterator over all leaf nodes in the mesh hierarchy.  
    ///
    /// This iterator allows traversal of the entire hierarchical mesh,  
    /// returning only the leaf nodes.  
    ///
    /// Example usage:
    /// 
    ///    let leaves: Vec<_> = node.leaf_iter().collect();  
    ///
    pub fn leaf_iter(&self) -> LeafIterator<T> {
        LeafIterator { stack: vec![self] }
    }
}

/// An iterator for traversing through leaf nodes in the hierarchical mesh.  
/// 
/// This iterator traverses all nodes in the hierarchy but only returns  
/// the leaf nodes.
pub struct LeafIterator<'a, T> {
    stack: Vec<&'a MeshNode<T>>,
}

impl<'a, T> Iterator for LeafIterator<'a, T> {
    type Item = &'a T;

    /// Returns the next leaf node in the traversal.  
    /// If the current node is a branch, its children are pushed onto the stack  
    /// for traversal in depth-first order.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match node {
                MeshNode::Leaf(data) => return Some(data),
                MeshNode::Branch { children, .. } => {
                    // Push children onto the stack in reverse order to get them in the desired order.
                    for child in children.iter().rev() {
                        self.stack.push(child);
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh::hierarchical::MeshNode;

    #[test]
    /// Test that verifies refining a leaf node into a branch works as expected.  
    fn test_refine_leaf() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        if let MeshNode::Branch { children, .. } = node {
            assert_eq!(children[0], MeshNode::Leaf(11));
            assert_eq!(children[1], MeshNode::Leaf(12));
            assert_eq!(children[2], MeshNode::Leaf(13));
            assert_eq!(children[3], MeshNode::Leaf(14));
        } else {
            panic!("Node should have been refined to a branch.");
        }
    }

    #[test]
    /// Test that verifies coarsening a branch node back into a leaf works as expected.  
    fn test_coarsen_branch() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);
        node.coarsen();

        assert_eq!(node, MeshNode::Leaf(10));
    }

    #[test]
    /// Test that verifies applying hanging node constraints works correctly by  
    /// averaging the degrees of freedom from the child elements to the parent element.  
    fn test_apply_hanging_node_constraints() {
        let node = MeshNode::Branch {
            data: 0,
            children: Box::new([
                MeshNode::Leaf(1),
                MeshNode::Leaf(2),
                MeshNode::Leaf(3),
                MeshNode::Leaf(4),
            ]),
        };

        let mut parent_dofs = [0.0; 4];
        let mut child_dofs = [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ];

        node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);

        assert_eq!(parent_dofs, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    /// Test that verifies the leaf iterator correctly traverses all leaf nodes in  
    /// the mesh hierarchy and returns them in the expected order.  
    fn test_leaf_iterator() {
        let mut node = MeshNode::Leaf(10);
        node.refine(|&data| [data + 1, data + 2, data + 3, data + 4]);

        let leaves: Vec<_> = node.leaf_iter().collect();
        assert_eq!(leaves, [&11, &12, &13, &14]);
    }
}
```

`src/domain/mesh/tests.rs`

```rust
#[cfg(test)]
mod tests {
    use crate::domain::mesh_entity::MeshEntity;
    use crossbeam::channel::unbounded;
    use crate::domain::mesh::Mesh;

    /// Tests that boundary data can be sent from one mesh and received by another.  
    /// This test sets up vertex coordinates for a mesh, sends the data,  
    /// and verifies that the receiving mesh gets the correct data.  
    #[test]
    fn test_send_receive_boundary_data() {
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        // Set up vertex coordinates.
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        mesh.vertex_coordinates.insert(2, [4.0, 5.0, 6.0]);

        // Add boundary entities.
        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);

        // Set up a separate sender and receiver for testing.
        let (test_sender, test_receiver) = unbounded();
        mesh.set_boundary_channels(test_sender, test_receiver);

        // Simulate sending the boundary data.
        mesh.send_boundary_data();

        // Create a second mesh instance to simulate the receiver.
        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(mesh.boundary_data_sender.clone().unwrap(), mesh.boundary_data_receiver.clone().unwrap());

        // Simulate receiving the boundary data.
        mesh_receiver.receive_boundary_data();

        // Verify that the receiver mesh has the updated vertex coordinates.
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[1.0, 2.0, 3.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[4.0, 5.0, 6.0]));
    }

    /// Tests that sending boundary data without a receiver does not cause a failure.  
    /// This ensures that missing a receiver does not result in panics or unexpected errors.  
    #[test]
    fn test_send_without_receiver() {
        let mut mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(3);
        mesh.vertex_coordinates.insert(3, [7.0, 8.0, 9.0]);
        mesh.add_entity(vertex);

        // Simulate sending the boundary data without setting a receiver.
        mesh.send_boundary_data();

        // No receiver to process, but this should not panic or fail.
        assert!(mesh.vertex_coordinates.get(&3).is_some());
    }

    /// Tests the addition of a new entity to the mesh.  
    /// Verifies that the entity is successfully added to the mesh's entity set.  
    #[test]
    fn test_add_entity() {
        let mesh = Mesh::new();
        let vertex = MeshEntity::Vertex(1);
        mesh.add_entity(vertex);
        assert!(mesh.entities.read().unwrap().contains(&vertex));
    }

    /// Tests the iterator over the mesh's vertex coordinates.  
    /// Verifies that the iterator returns the correct vertex IDs.  
    #[test]
    fn test_iter_vertices() {
        let mut mesh = Mesh::new();
        mesh.vertex_coordinates.insert(1, [1.0, 2.0, 3.0]);
        let vertices: Vec<_> = mesh.iter_vertices().collect();
        assert_eq!(vertices, vec![&1]);
    }
}

#[cfg(test)]
mod integration_tests {
    use crate::domain::mesh::hierarchical::MeshNode;
    use crate::domain::mesh_entity::MeshEntity;
    use crate::domain::mesh::Mesh;
    use crossbeam::channel::unbounded;

    /// Full integration test that simulates mesh operations including entity addition,  
    /// boundary data synchronization, hierarchical mesh refinement, and applying  
    /// constraints at hanging nodes.  
    #[test]
    fn test_full_mesh_integration() {
        // Step 1: Create a new mesh and add entities (vertices, edges, cells)
        let mut mesh = Mesh::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);
        let cell1 = MeshEntity::Cell(1);

        mesh.add_entity(vertex1);
        mesh.add_entity(vertex2);
        mesh.add_entity(vertex3);
        mesh.add_entity(cell1);
        mesh.set_vertex_coordinates(1, [0.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(2, [1.0, 0.0, 0.0]);
        mesh.set_vertex_coordinates(3, [0.0, 1.0, 0.0]);

        // Step 2: Set up and sync boundary data.
        let (sender, receiver) = unbounded();
        mesh.set_boundary_channels(sender, receiver);
        mesh.send_boundary_data();

        // Create another mesh instance to simulate receiving boundary data.
        let mut mesh_receiver = Mesh::new();
        mesh_receiver.set_boundary_channels(mesh.boundary_data_sender.clone().unwrap(), mesh.boundary_data_receiver.clone().unwrap());
        mesh_receiver.receive_boundary_data();

        // Verify that the receiver mesh has the correct boundary data.
        assert_eq!(mesh_receiver.vertex_coordinates.get(&1), Some(&[0.0, 0.0, 0.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&2), Some(&[1.0, 0.0, 0.0]));
        assert_eq!(mesh_receiver.vertex_coordinates.get(&3), Some(&[0.0, 1.0, 0.0]));

        // Step 3: Refine a hierarchical mesh node.
        let mut node = MeshNode::Leaf(cell1);
        node.refine(|&_cell| [
            MeshEntity::Cell(2), MeshEntity::Cell(3), MeshEntity::Cell(4), MeshEntity::Cell(5)
        ]);

        // Verify that the node has been refined into a branch.
        if let MeshNode::Branch { ref children, .. } = node {
            assert_eq!(children.len(), 4);
            assert_eq!(children[0], MeshNode::Leaf(MeshEntity::Cell(2)));
            assert_eq!(children[1], MeshNode::Leaf(MeshEntity::Cell(3)));
            assert_eq!(children[2], MeshNode::Leaf(MeshEntity::Cell(4)));
            assert_eq!(children[3], MeshNode::Leaf(MeshEntity::Cell(5)));
        } else {
            panic!("Expected the node to be refined into a branch.");
        }

        // Step 4: Apply RCM ordering to the mesh and verify order.
        let rcm_order = mesh.rcm_ordering(vertex1);
        assert!(rcm_order.len() > 0); // RCM ordering should produce a non-empty order.

        // Step 5: Apply constraints at the hanging nodes after refinement.
        let mut parent_dofs = [0.0; 4];
        let mut child_dofs = [
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
            [1.0, 1.5, 2.0, 2.5],
        ];
        node.apply_hanging_node_constraints(&mut parent_dofs, &mut child_dofs);

        // Verify that the hanging node constraints were applied correctly.
        assert_eq!(parent_dofs, [1.0, 1.5, 2.0, 2.5]);
    }
}
```

`src/domain/mod.rs`

```rust


pub mod mesh_entity;
pub mod sieve;
pub mod section;
pub mod overlap;
pub mod stratify;
pub mod entity_fill;
pub mod mesh;

/// Re-exports key components from the `mesh_entity`, `sieve`, and `section` modules.  
/// 
/// This allows the user to access the `MeshEntity`, `Arrow`, `Sieve`, and `Section`  
/// structs directly when importing this module.  
///
/// Example usage:
///
///    use crate::domain::{MeshEntity, Arrow, Sieve, Section};  
///    let entity = MeshEntity::Vertex(1);  
///    let sieve = Sieve::new();  
///    let section: Section<f64> = Section::new();  
/// 
pub use mesh_entity::{MeshEntity, Arrow};
pub use sieve::Sieve;
pub use section::Section;
```

`src/domain/overlap.rs`

```rust
use dashmap::DashMap;
use std::sync::Arc;
use crate::domain::mesh_entity::MeshEntity;

/// The `Overlap` struct manages two sets of `MeshEntity` elements:  
/// - `local_entities`: Entities that are local to the current partition.
/// - `ghost_entities`: Entities that are shared with other partitions.
pub struct Overlap {
    /// A thread-safe set of local entities.  
    pub local_entities: Arc<DashMap<MeshEntity, ()>>,
    /// A thread-safe set of ghost entities.  
    pub ghost_entities: Arc<DashMap<MeshEntity, ()>>,
}

impl Overlap {
    /// Creates a new `Overlap` with empty sets for local and ghost entities.
    pub fn new() -> Self {
        Overlap {
            local_entities: Arc::new(DashMap::new()),
            ghost_entities: Arc::new(DashMap::new()),
        }
    }

    /// Adds a `MeshEntity` to the set of local entities.
    pub fn add_local_entity(&self, entity: MeshEntity) {
        self.local_entities.insert(entity, ());
    }

    /// Adds a `MeshEntity` to the set of ghost entities.
    pub fn add_ghost_entity(&self, entity: MeshEntity) {
        self.ghost_entities.insert(entity, ());
    }

    /// Checks if a `MeshEntity` is a local entity.
    pub fn is_local(&self, entity: &MeshEntity) -> bool {
        self.local_entities.contains_key(entity)
    }

    /// Checks if a `MeshEntity` is a ghost entity.
    pub fn is_ghost(&self, entity: &MeshEntity) -> bool {
        self.ghost_entities.contains_key(entity)
    }

    /// Retrieves a clone of all local entities.
    pub fn local_entities(&self) -> Vec<MeshEntity> {
        self.local_entities.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves a clone of all ghost entities.
    pub fn ghost_entities(&self) -> Vec<MeshEntity> {
        self.ghost_entities.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Merges another `Overlap` instance into this one, combining local  
    /// and ghost entities from both overlaps.
    pub fn merge(&self, other: &Overlap) {
        other.local_entities.iter().for_each(|entry| {
            self.local_entities.insert(entry.key().clone(), ());
        });

        other.ghost_entities.iter().for_each(|entry| {
            self.ghost_entities.insert(entry.key().clone(), ());
        });
    }
}

/// The `Delta` struct manages transformation data for `MeshEntity` elements  
/// in overlapping regions. It is used to store and apply data transformations  
/// across entities in distributed environments.
pub struct Delta<T> {
    /// A thread-safe map storing transformation data associated with `MeshEntity` objects.  
    pub data: Arc<DashMap<MeshEntity, T>>,  // Transformation data over overlapping regions
}

impl<T> Delta<T> {
    /// Creates a new, empty `Delta`.
    pub fn new() -> Self {
        Delta {
            data: Arc::new(DashMap::new()),
        }
    }

    /// Sets the transformation data for a specific `MeshEntity`.
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Retrieves the transformation data associated with a specific `MeshEntity`.
    pub fn get_data(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|entry| entry.clone())
    }

    /// Removes the transformation data associated with a specific `MeshEntity`.
    pub fn remove_data(&self, entity: &MeshEntity) -> Option<T> {
        self.data.remove(entity).map(|(_, value)| value)
    }

    /// Checks if there is transformation data for a specific `MeshEntity`.
    pub fn has_data(&self, entity: &MeshEntity) -> bool {
        self.data.contains_key(entity)
    }

    /// Applies a function to all entities in the delta.
    pub fn apply<F>(&self, mut func: F)
    where
        F: FnMut(&MeshEntity, &T),
    {
        self.data.iter().for_each(|entry| func(entry.key(), entry.value()));
    }

    /// Merges another `Delta` instance into this one, combining data from both deltas.
    pub fn merge(&self, other: &Delta<T>)
    where
        T: Clone,
    {
        other.data.iter().for_each(|entry| {
            self.data.insert(entry.key().clone(), entry.value().clone());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    fn test_overlap_local_and_ghost_entities() {
        let overlap = Overlap::new();
        let vertex_local = MeshEntity::Vertex(1);
        let vertex_ghost = MeshEntity::Vertex(2);
        overlap.add_local_entity(vertex_local);
        overlap.add_ghost_entity(vertex_ghost);
        assert!(overlap.is_local(&vertex_local));
        assert!(overlap.is_ghost(&vertex_ghost));
    }

    #[test]
    fn test_overlap_merge() {
        let overlap1 = Overlap::new();
        let overlap2 = Overlap::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);
        let vertex3 = MeshEntity::Vertex(3);

        overlap1.add_local_entity(vertex1);
        overlap1.add_ghost_entity(vertex2);

        overlap2.add_local_entity(vertex3);

        overlap1.merge(&overlap2);

        assert!(overlap1.is_local(&vertex1));
        assert!(overlap1.is_ghost(&vertex2));
        assert!(overlap1.is_local(&vertex3));
        assert_eq!(overlap1.local_entities().len(), 2);
    }

    #[test]
    fn test_delta_set_and_get_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 42);

        assert_eq!(delta.get_data(&vertex), Some(42));
        assert!(delta.has_data(&vertex));
    }

    #[test]
    fn test_delta_remove_data() {
        let delta = Delta::new();
        let vertex = MeshEntity::Vertex(1);

        delta.set_data(vertex, 100);
        assert_eq!(delta.remove_data(&vertex), Some(100));
        assert!(!delta.has_data(&vertex));
    }

    #[test]
    fn test_delta_merge() {
        let delta1 = Delta::new();
        let delta2 = Delta::new();
        let vertex1 = MeshEntity::Vertex(1);
        let vertex2 = MeshEntity::Vertex(2);

        delta1.set_data(vertex1, 10);
        delta2.set_data(vertex2, 20);

        delta1.merge(&delta2);

        assert_eq!(delta1.get_data(&vertex1), Some(10));
        assert_eq!(delta1.get_data(&vertex2), Some(20));
    }
}
```

`src/domain/section.rs`

```rust
use dashmap::DashMap;
use rayon::prelude::*;
use crate::domain::mesh_entity::MeshEntity;

/// A generic `Section` struct that associates data of type `T` with `MeshEntity` elements.  
/// It provides methods for setting, updating, and retrieving data, and supports  
/// parallel updates for performance improvements.  
///
/// Example usage:
///
///    let section = Section::new();  
///    let vertex = MeshEntity::Vertex(1);  
///    section.set_data(vertex, 42);  
///    assert_eq!(section.restrict(&vertex), Some(42));  
/// 
pub struct Section<T> {
    /// A thread-safe map storing data of type `T` associated with `MeshEntity` objects.  
    pub data: DashMap<MeshEntity, T>,
}

impl<T> Section<T> {
    /// Creates a new `Section` with an empty data map.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    assert!(section.data.is_empty());  
    ///
    pub fn new() -> Self {
        Section {
            data: DashMap::new(),
        }
    }

    /// Sets the data associated with a given `MeshEntity`.  
    /// This method inserts the `entity` and its corresponding `value` into the data map.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    section.set_data(MeshEntity::Vertex(1), 10);  
    ///
    pub fn set_data(&self, entity: MeshEntity, value: T) {
        self.data.insert(entity, value);
    }

    /// Restricts the data for a given `MeshEntity` by returning an immutable copy of the data  
    /// associated with the `entity`, if it exists.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    section.set_data(vertex, 42);  
    ///    assert_eq!(section.restrict(&vertex), Some(42));  
    ///
    pub fn restrict(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Applies the given function in parallel to update all data values in the section.
    ///
    /// Example usage:
    ///
    ///    section.parallel_update(|v| *v += 1);  
    ///
    pub fn parallel_update<F>(&self, update_fn: F)
    where
        F: Fn(&mut T) + Sync + Send,
        T: Send + Sync,
    {
        // Clone the keys to ensure safe access to each mutable entry in parallel.
        let keys: Vec<MeshEntity> = self.data.iter().map(|entry| entry.key().clone()).collect();

        // Apply the update function to each entry in parallel.
        keys.into_par_iter().for_each(|key| {
            if let Some(mut entry) = self.data.get_mut(&key) {
                update_fn(entry.value_mut());
            }
        });
    }

    /// Restricts the data for a given `MeshEntity` by returning a mutable copy of the data  
    /// associated with the `entity`, if it exists.  
    ///
    /// Returns `None` if no data is found for the entity.  
    ///
    /// Example usage:
    ///
    ///    let section = Section::new();  
    ///    let vertex = MeshEntity::Vertex(1);  
    ///    section.set_data(vertex, 5);  
    ///    let mut value = section.restrict_mut(&vertex).unwrap();  
    ///    value = 10;  
    ///    section.set_data(vertex, value);  
    ///
    pub fn restrict_mut(&self, entity: &MeshEntity) -> Option<T>
    where
        T: Clone,
    {
        self.data.get(entity).map(|v| v.clone())
    }

    /// Updates the data for a specific `MeshEntity` by replacing the existing value  
    /// with the new value.  
    ///
    /// Example usage:
    ///
    ///    section.update_data(&MeshEntity::Vertex(1), 15);  
    ///
    pub fn update_data(&self, entity: &MeshEntity, new_value: T) {
        self.data.insert(*entity, new_value);
    }

    /// Clears all data from the section, removing all entity associations.  
    ///
    /// Example usage:
    ///
    ///    section.clear();  
    ///    assert!(section.data.is_empty());  
    ///
    pub fn clear(&self) {
        self.data.clear();
    }

    /// Retrieves all `MeshEntity` objects associated with the section.  
    ///
    /// Returns a vector containing all mesh entities currently stored in the section.  
    ///
    /// Example usage:
    ///
    ///    let entities = section.entities();  
    ///
    pub fn entities(&self) -> Vec<MeshEntity> {
        self.data.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Retrieves all data stored in the section as immutable copies.  
    ///
    /// Returns a vector of data values.  
    ///
    /// Example usage:
    ///
    ///    let all_data = section.all_data();  
    ///
    pub fn all_data(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().map(|entry| entry.value().clone()).collect()
    }

    /// Retrieves all data stored in the section with mutable access.  
    ///
    /// Returns a vector of data values that can be modified.  
    ///
    /// Example usage:
    ///
    ///    let all_data_mut = section.all_data_mut();  
    ///
    pub fn all_data_mut(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter_mut().map(|entry| entry.value().clone()).collect()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::mesh_entity::MeshEntity;

    #[test]
    /// Test that verifies setting and restricting data for a `MeshEntity`  
    /// works as expected.  
    fn test_set_and_restrict_data() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        section.set_data(vertex, 42);
        assert_eq!(section.restrict(&vertex), Some(42));
    }

    #[test]
    /// Test that verifies updating the data for an entity works as expected,  
    /// including updating a non-existent entity.  
    fn test_update_data() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);

        section.set_data(vertex, 10);
        assert_eq!(section.restrict(&vertex), Some(10));

        // Update the data
        section.update_data(&vertex, 15);
        assert_eq!(section.restrict(&vertex), Some(15));

        // Try updating data for a non-existent entity (should insert it)
        let non_existent_entity = MeshEntity::Vertex(2);
        section.update_data(&non_existent_entity, 30);
        assert_eq!(section.restrict(&non_existent_entity), Some(30));
    }

    #[test]
    /// Test that verifies the mutable restriction of data for a `MeshEntity`  
    /// works as expected.  
    fn test_restrict_mut() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);

        section.set_data(vertex, 5);
        if let Some(mut value) = section.restrict_mut(&vertex) {
            value = 50;
            section.set_data(vertex, value);
        }
        assert_eq!(section.restrict(&vertex), Some(50));
    }

    #[test]
    /// Test that verifies retrieving all entities associated with the section  
    /// works as expected.  
    fn test_get_all_entities() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        section.set_data(vertex, 10);
        section.set_data(edge, 20);

        let entities = section.entities();
        assert!(entities.contains(&vertex));
        assert!(entities.contains(&edge));
        assert_eq!(entities.len(), 2);
    }

    #[test]
    /// Test that verifies retrieving all data stored in the section works  
    /// as expected.  
    fn test_get_all_data() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        let edge = MeshEntity::Edge(1);

        section.set_data(vertex, 10);
        section.set_data(edge, 20);

        let all_data = section.all_data();
        assert_eq!(all_data.len(), 2);
        assert!(all_data.contains(&10));
        assert!(all_data.contains(&20));
    }

    #[test]
    /// Test that verifies parallel updates to data in the section are  
    /// applied correctly using Rayon for concurrency.  
    fn test_parallel_update() {
        let section = Section::new();
        let vertex = MeshEntity::Vertex(1);
        section.set_data(vertex, 10);
        section.parallel_update(|v| *v += 5);
        assert_eq!(section.restrict(&vertex), Some(15));
    }
}
```

`src/domain/entity_fill.rs`

```rust
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use dashmap::DashMap;

impl Sieve {
    /// Infers and adds missing edges (in 2D) or faces (in 3D) based on existing cells and vertices.  
    /// 
    /// For 2D meshes, this method generates edges by connecting vertices of a cell.  
    /// These edges are then associated with the corresponding vertices in the sieve.  
    ///
    /// Example usage:
    /// 
    ///    sieve.fill_missing_entities();  
    ///
    pub fn fill_missing_entities(&self) {
        // Use DashMap instead of FxHashSet for concurrent access.
        let edge_set: DashMap<(MeshEntity, MeshEntity), ()> = DashMap::new();

        // Loop through each cell and infer its edges (for 2D meshes)
        self.adjacency.iter().for_each(|entry| {
            let cell = entry.key();
            if let MeshEntity::Cell(_) = cell {
                let vertices: Vec<_> = entry.value().iter().map(|v| v.key().clone()).collect();
                // Connect each vertex with its neighboring vertex to form edges.
                for i in 0..vertices.len() {
                    let v1 = vertices[i].clone();
                    let v2 = vertices[(i + 1) % vertices.len()].clone();
                    let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
                    edge_set.insert(edge, ());
                }
            }
        });

        // Add the deduced edges to the sieve.
        let edge_count = self.adjacency.len();
        edge_set.into_iter().enumerate().for_each(|(index, ((v1, v2), _))| {
            // Generate a unique ID for the new edge.
            let edge = MeshEntity::Edge(edge_count + index);
            self.add_arrow(v1, edge.clone());
            self.add_arrow(v2, edge);
        });
    }
}
```

`src/domain/stratify.rs`

```rust
use crate::domain::mesh_entity::MeshEntity;
use crate::domain::sieve::Sieve;
use dashmap::DashMap;

/// Implements a stratification method for the `Sieve` structure.  
/// Stratification organizes the mesh entities into different strata based on  
/// their dimensions:  
/// - Stratum 0: Vertices  
/// - Stratum 1: Edges  
/// - Stratum 2: Faces  
/// - Stratum 3: Cells  
///
/// This method categorizes each `MeshEntity` into its corresponding stratum and  
/// returns a `DashMap` where the keys are the dimension (stratum) and the values  
/// are vectors of mesh entities in that stratum.  
///
/// Example usage:
/// 
///    let sieve = Sieve::new();  
///    sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));  
///    let strata = sieve.stratify();  
///    assert_eq!(strata.get(&0).unwrap().len(), 1);  // Stratum for vertices  
/// 
impl Sieve {
    /// Organizes the mesh entities in the sieve into strata based on their dimension.  
    ///
    /// The method creates a map where each key is the dimension (0 for vertices,  
    /// 1 for edges, 2 for faces, 3 for cells), and the value is a vector of mesh  
    /// entities in that dimension.
    ///
    /// Example usage:
    /// 
    ///    let sieve = Sieve::new();  
    ///    sieve.add_arrow(MeshEntity::Vertex(1), MeshEntity::Edge(1));  
    ///    let strata = sieve.stratify();  
    ///
    pub fn stratify(&self) -> DashMap<usize, Vec<MeshEntity>> {
        let strata: DashMap<usize, Vec<MeshEntity>> = DashMap::new();

        // Iterate over the adjacency map to classify entities by their dimension.
        self.adjacency.iter().for_each(|entry| {
            let entity = entry.key();
            // Determine the dimension of the current entity.
            let dimension = match entity {
                MeshEntity::Vertex(_) => 0,  // Stratum 0 for vertices
                MeshEntity::Edge(_) => 1,    // Stratum 1 for edges
                MeshEntity::Face(_) => 2,    // Stratum 2 for faces
                MeshEntity::Cell(_) => 3,    // Stratum 3 for cells
            };
            
            // Insert entity into the appropriate stratum in a thread-safe manner.
            strata.entry(dimension).or_insert_with(Vec::new).push(entity.clone());
        });

        strata
    }
}
```

This is all of the components of the Hydra `Domain` module. Please now prepare the outline discussed at the beginning of this conversation.
