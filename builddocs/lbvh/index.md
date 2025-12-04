# LBVH

In order to implement "closest point to triangular mesh" queries and similar, an LBVH is implemented, the test for which generates the LBVH visualized in the following section. 

The implementation follows ["Maximizing Parallelism in the Construction of BVHs,
Octrees, and k-d Trees"](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf) by Tero Karras and the accompanying blog post [Thinking Parallel, Part III: Tree Construction on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/), and comparison with [Toru Niina's implementation](https://github.com/ToruNiina/lbvh/tree/master) during debugging.

## Visualization
For the Stanford dragon model, an LBVH is built, its leaf and internal nodes visualized and "closest point on mesh" queries are tested by projecting randomly sampled points onto the mesh.

<div class="checkbox-container">
    <div>
        <input type="checkbox" id="show_leaf" name="Show Leaf Nodes" class="input-elem">
            <span class="input-label">Show Leaf Nodes</span>
        </input>
    </div>
    <div>
        <input type="checkbox" id="show_internal" name="Show Internal Nodes" class="input-elem">
            <span class="input-label">Show Internal Nodes</span>
        </input>
    </div>
    <div>
        <input type="checkbox" id="show_mesh" name="Show Mesh" checked class="input-elem">
            <span class="input-label">Show Mesh</span>
        </input>
    </div>
    <div>
        <input type="checkbox" id="show_original" name="Show Points" checked class="input-elem">
            <span class="input-label">Show Original Points</span>
        </input>
    </div>
    <div>
        <input type="checkbox" id="show_projected" name="Show Projected Points" checked class="input-elem">
            <span class="input-label">Show Projected Points</span>
        </input>
    </div>
    <div>
        <input type="checkbox" id="show_connections" name="Show Connections" checked class="input-elem">
            <span class="input-label">Show Connections</span>
        </input>
    </div>
</div>

<div id="lbvhboxes"></div>

<script type="importmap">
{
    "imports": {
        "three": "https://cdn.jsdelivr.net/npm/three@0.181.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.181.0/examples/jsm/"
    }
}
</script>

<style>
#lbvhboxes{  
    width: 100%;
    height: max(500px, 80vh);
    position: relative;
}
#lbvhboxes > div{  
    position: absolute !important;
}
.checkbox-container{
    border: 2px solid var(--color-code-foreground);
    padding:20px;  
    margin: 20px;
    border-radius:5px;  
    display: flex;
    flex-direction: row;  
    width: calc(100% - 80px);
    flex-wrap: wrap;
}
.input-label{
    margin-left:10px;
}
.input-elem{
    width: 30px; 
    height: 30px;
    margin-left: 10px;
}
</style>
<script type="module">
    import * as THREE from 'three';

    import Stats from 'three/addons/libs/stats.module.js';
    import { OBJLoader } from 'three/addons/loaders/OBJLoader.js'
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

    let container = document.getElementById("lbvhboxes")
    let camera = new THREE.PerspectiveCamera(50, 0.5*container.clientWidth / container.clientHeight, 0.01, 100);
    camera.position.z = 5;
    let controls = new OrbitControls(camera, container)
    controls.enableDamping = true

    let stats = new Stats();

    let scene, renderer;
    let onWindowResize = () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    scene = new THREE.Scene();

    const ambientLight = new THREE.AmbientLight(0xffffff, 1)
    scene.add(ambientLight)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9)
    directionalLight.position.set(1, 0.25, 0)
    scene.add(directionalLight)

    const bvh_int_material = new THREE.MeshBasicMaterial({
        color: "royalblue", 
        opacity: 0.1, 
        transparent: true, 
        depthTest: true,
        side: THREE.DoubleSide
    })
    const bvh_leaf_material = new THREE.MeshStandardMaterial({
        color: "white", 
        opacity: 0.5, 
        transparent: true, 
        depthTest: true,
        side: THREE.DoubleSide
    })


    // instanced AABB rendering: leafs
    const res = await fetch('../_static/lbvh_leafs.bin');
    const buf = await res.arrayBuffer();
    const leafs = new Float32Array(buf);
    const N = leafs.length / 6
    const leaf_instance_mesh = new THREE.InstancedMesh(
        new THREE.BoxGeometry(1,1,1), 
        bvh_leaf_material, 
        N
    );
    scene.add(leaf_instance_mesh);
    const dummy = new THREE.Object3D();
    for (let i = 0; i < N; i++) {
        const px = leafs[i*6 + 0];
        const py = leafs[i*6 + 1];
        const pz = leafs[i*6 + 2];
        const sx = leafs[i*6 + 3];
        const sy = leafs[i*6 + 4];
        const sz = leafs[i*6 + 5];
        dummy.position.set(px, py, pz);
        dummy.scale.set(sx, sy, sz);
        dummy.updateMatrix();
        leaf_instance_mesh.setMatrixAt(i, dummy.matrix);
    }

    // instanced AABB rendering: internal nodes
    const res_int = await fetch('../_static/lbvh_internals.bin');
    const buf_int = await res_int.arrayBuffer();
    const internals = new Float32Array(buf_int);
    const N_int = leafs.length / 6
    const internal_instance_mesh = new THREE.InstancedMesh(
        new THREE.BoxGeometry(1,1,1), 
        bvh_int_material, 
        N
    );
    // console.log(internals, N_int)
    scene.add(internal_instance_mesh);
    for (let i = 0; i < N_int; i++) {
        const px = internals[i*6 + 0];
        const py = internals[i*6 + 1];
        const pz = internals[i*6 + 2];
        const sx = internals[i*6 + 3];
        const sy = internals[i*6 + 4];
        const sz = internals[i*6 + 5];
        dummy.position.set(px, py, pz);
        dummy.scale.set(sx, sy, sz);
        dummy.updateMatrix();
        internal_instance_mesh.setMatrixAt(i, dummy.matrix);
    }

    // show point clouds
    const points_to_scene = async (points_path, colour, size=0.02) => {
        const res_pp = await fetch(points_path);
        const buf_pp = await res_pp.arrayBuffer();
        const proj_points_buf = new Float32Array(buf_pp);
        const N_points = proj_points_buf.length / 3
        const proj_points_geo = new THREE.BufferGeometry();
        const proj_verts = []
        for (let i = 0; i < N_points; i++) {
            proj_verts.push(
                proj_points_buf[i*3+0],
                proj_points_buf[i*3+1],
                proj_points_buf[i*3+2]
            );
        }
        proj_points_geo.setAttribute( 'position', new THREE.Float32BufferAttribute( proj_verts, 3 ) );
        const proj_points_mat = new THREE.PointsMaterial({
                color: colour,
                size: size,
                sizeAttenuation: true
            });
        return new THREE.Points(proj_points_geo, proj_points_mat)
    }
    const points_proj = await points_to_scene('../_static/points_proj.bin', "orange", 0.05)
    const points_orig = await points_to_scene('../_static/points_orig.bin', "white")
    scene.add(points_proj);
    scene.add(points_orig);

    // add connecting lines
    const res_lines_orig = await fetch('../_static/points_orig.bin');
    const buf_lines_orig = await res_lines_orig.arrayBuffer();
    const orig_buf = new Float32Array(buf_lines_orig);
    const res_lines_proj = await fetch('../_static/points_proj.bin');
    const buf_lines_proj = await res_lines_proj.arrayBuffer();
    const proj_buf = new Float32Array(buf_lines_proj);
    const N_lines = Math.min(orig_buf.length, proj_buf.length) / 3;
    const line_positions = new Float32Array(N_lines * 6); // 2 vertices per segment
    for (let i = 0; i < N_lines; i++) {
        // original point
        line_positions[i*6 + 0] = orig_buf[i*3 + 0];
        line_positions[i*6 + 1] = orig_buf[i*3 + 1];
        line_positions[i*6 + 2] = orig_buf[i*3 + 2];
        // projected point
        line_positions[i*6 + 3] = proj_buf[i*3 + 0];
        line_positions[i*6 + 4] = proj_buf[i*3 + 1];
        line_positions[i*6 + 5] = proj_buf[i*3 + 2];
    }

    const connections_geo = new THREE.BufferGeometry();
    connections_geo.setAttribute('position', new THREE.Float32BufferAttribute(line_positions, 3));
    const connections_mat = new THREE.LineBasicMaterial({
        color: 'yellow',
        transparent: true,
        opacity: 0.9,
        depthTest: true
    });
    const connection_lines = new THREE.LineSegments(connections_geo, connections_mat);
    scene.add(connection_lines);


    // load mesh
    const material =  new THREE.MeshStandardMaterial({color: "white", opacity: 0.5, transparent: true, side: THREE.DoubleSide})
    const loader = new OBJLoader();
    const model = await loader.loadAsync( '../_static/dragon.obj'  );
    scene.add( model );

    let animate = () => {
        controls.update()
        renderer.render(scene, camera);
        stats.update();
        leaf_instance_mesh.visible = document.getElementById("show_leaf")?.checked
        internal_instance_mesh.visible = document.getElementById("show_internal")?.checked
        model.visible = document.getElementById("show_mesh")?.checked
        points_proj.visible = document.getElementById("show_projected")?.checked
        points_orig.visible = document.getElementById("show_original")?.checked
        connection_lines.visible = document.getElementById("show_connections")?.checked
    }
    renderer = new THREE.WebGLRenderer({
        logarithmicDepthBuffer: true 
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setAnimationLoop(animate);
    window.addEventListener('resize', onWindowResize);
    container.appendChild(renderer.domElement);
    onWindowResize()

    // show stats
    container.appendChild(stats.dom);
</script>
