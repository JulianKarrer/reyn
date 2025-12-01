# LBVH

In order to implement "closest point to triangular mesh" queries and similar, an LBVH is implemented, the test for which generates the LBVH visualized in the following section. 

The implementation follows ["Maximizing Parallelism in the Construction of BVHs,
Octrees, and k-d Trees"](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf) by Tero Karras and the accompanying blog post [Thinking Parallel, Part III: Tree Construction on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/), while ocassionally peeking at [Toru Niina's implementation](https://github.com/ToruNiina/lbvh/tree/master) during debugging.

## Visualization

<div class="checkbox-container">
    <input type="checkbox" id="show_leaf" name="Show Leaf Nodes" checked style="width: 30px; height: 30px;">
        <span style="margin-left: 20px;">Show Leaf Nodes</span>
    </input>
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
}
</style>
<script type="module">
    import * as THREE from 'three';

    import Stats from 'three/addons/libs/stats.module.js';
    // import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
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
        opacity: 0.9, 
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
    console.log(internals, N_int)
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

    // load mesh
    // const material =  new THREE.MeshStandardMaterial({color: "white"})
    // const loader = new PLYLoader();
    // const geometry = await loader.loadAsync( '../_static/dragon.ply'  );
    // scene.add( new THREE.Mesh( geometry ) );

    let animate = () => {
        controls.update()
        renderer.render(scene, camera);
        stats.update();
        leaf_instance_mesh.visible = document.getElementById("show_leaf")?.checked
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
