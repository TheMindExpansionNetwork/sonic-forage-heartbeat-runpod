import * as THREE from 'three/webgpu';
import { pass, mrt, output, normalView, diffuseColor, velocity, add, vec4, directionToColor, colorToDirection, sample } from 'three/tsl';
import { ssgi } from 'three/addons/tsl/display/SSGINode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

// MediaPipe (face tracking)
import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';

// ── Config ──

const BOX_HEIGHT = 6;
const BOX_DEPTH = 8;
const WALL_THICKNESS = 0.5;
const CAM_FOV = 45;
const FACE_FILL = 0.62; // fraction of box height the face should occupy
const LIGHT_DEPTH = BOX_DEPTH / 2 + 2; // point-light plane, forward toward the camera
const DEMON_PALETTE = {
	bg: 0x06060c,
	frame: 0x0d0d14,
	panel: 0x11111a,
	teal: 0x3db6be,
	mustard: 0xc7b566,
	orange: 0xf08a48,
	coral: 0xe84f3d,
	clay: 0xd6d6e0,
};

function getBoxWidth() {

	const aspect = window.innerWidth / window.innerHeight;
	const vFov = THREE.MathUtils.degToRad( CAM_FOV / 2 );
	// Width that fills the viewport at the front face distance
	const dist = ( BOX_HEIGHT / 2 ) / Math.tan( vFov );
	return Math.tan( vFov ) * aspect * dist * 2;

}

// MediaPipe blendshape name → facecap morph target name
const blendshapesMap = {
	// '_neutral': '',
	'browDownLeft': 'browDown_L',
	'browDownRight': 'browDown_R',
	'browInnerUp': 'browInnerUp',
	'browOuterUpLeft': 'browOuterUp_L',
	'browOuterUpRight': 'browOuterUp_R',
	'cheekPuff': 'cheekPuff',
	'cheekSquintLeft': 'cheekSquint_L',
	'cheekSquintRight': 'cheekSquint_R',
	'eyeBlinkLeft': 'eyeBlink_L',
	'eyeBlinkRight': 'eyeBlink_R',
	'eyeLookDownLeft': 'eyeLookDown_L',
	'eyeLookDownRight': 'eyeLookDown_R',
	'eyeLookInLeft': 'eyeLookIn_L',
	'eyeLookInRight': 'eyeLookIn_R',
	'eyeLookOutLeft': 'eyeLookOut_L',
	'eyeLookOutRight': 'eyeLookOut_R',
	'eyeLookUpLeft': 'eyeLookUp_L',
	'eyeLookUpRight': 'eyeLookUp_R',
	'eyeSquintLeft': 'eyeSquint_L',
	'eyeSquintRight': 'eyeSquint_R',
	'eyeWideLeft': 'eyeWide_L',
	'eyeWideRight': 'eyeWide_R',
	'jawForward': 'jawForward',
	'jawLeft': 'jawLeft',
	'jawOpen': 'jawOpen',
	'jawRight': 'jawRight',
	'mouthClose': 'mouthClose',
	'mouthDimpleLeft': 'mouthDimple_L',
	'mouthDimpleRight': 'mouthDimple_R',
	'mouthFrownLeft': 'mouthFrown_L',
	'mouthFrownRight': 'mouthFrown_R',
	'mouthFunnel': 'mouthFunnel',
	'mouthLeft': 'mouthLeft',
	'mouthLowerDownLeft': 'mouthLowerDown_L',
	'mouthLowerDownRight': 'mouthLowerDown_R',
	'mouthPressLeft': 'mouthPress_L',
	'mouthPressRight': 'mouthPress_R',
	'mouthPucker': 'mouthPucker',
	'mouthRight': 'mouthRight',
	'mouthRollLower': 'mouthRollLower',
	'mouthRollUpper': 'mouthRollUpper',
	'mouthShrugLower': 'mouthShrugLower',
	'mouthShrugUpper': 'mouthShrugUpper',
	'mouthSmileLeft': 'mouthSmile_L',
	'mouthSmileRight': 'mouthSmile_R',
	'mouthStretchLeft': 'mouthStretch_L',
	'mouthStretchRight': 'mouthStretch_R',
	'mouthUpperUpLeft': 'mouthUpperUp_L',
	'mouthUpperUpRight': 'mouthUpperUp_R',
	'noseSneerLeft': 'noseSneer_L',
	'noseSneerRight': 'noseSneer_R',
	// '': 'tongueOut'
};

// ── Three.js ──

let camera, scene, renderer, renderPipeline;
let raycaster, pointer;
let mouseLight, rimLight, lowLight;

// ── Cornell box ──

let wallMeshes = [];
let boxSize = { w: 8, h: BOX_HEIGHT, d: BOX_DEPTH };

// ── Face ──

let face, eyeL, eyeR, faceTransform, faceMaterial;
let faceRoot, faceBaseSize, faceBaseCenter, faceBaseScale, faceFitScale = 1;
let faceLandmarker, video;
const eyeRotationLimit = THREE.MathUtils.degToRad( 30 );
const transform = new THREE.Object3D();

// ── Mouse light easing ──

let mouseLightTarget = new THREE.Vector3( 0, BOX_HEIGHT / 2, LIGHT_DEPTH );
const EASE_SPEED = 8;
let targetBloom = 0;
let audioBloom = 0;

window.addEventListener( 'message', ( event ) => {

	if ( event.origin !== window.location.origin ) return;
	if ( event.data?.type !== 'demon:face-reactivity' ) return;
	targetBloom = THREE.MathUtils.clamp( Number( event.data.bloom ) || 0, 0, 1 );

} );

// ── Init ──

init();

async function init() {

	// Camera

	camera = new THREE.PerspectiveCamera( CAM_FOV, window.innerWidth / window.innerHeight, 0.1, 100 );

	// Scene

	scene = new THREE.Scene();
	scene.background = new THREE.Color( DEMON_PALETTE.bg );
	scene.fog = new THREE.Fog( DEMON_PALETTE.bg, 7, 17 );

	// Renderer

	renderer = new THREE.WebGPURenderer( { antialias: false } );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 0.42;
	renderer.shadowMap.enabled = true;
	document.body.appendChild( renderer.domElement );

	// WebGPU needs the device ready before KTX2Loader.detectSupport()
	await renderer.init();

	// Post-processing: SSGI

	renderPipeline = new THREE.RenderPipeline( renderer );

	const scenePass = pass( scene, camera );
	scenePass.setMRT( mrt( {
		output: output,
		diffuseColor: diffuseColor,
		normal: directionToColor( normalView ),
		velocity: velocity,
	} ) );

	const scenePassColor = scenePass.getTextureNode( 'output' );
	const scenePassDiffuse = scenePass.getTextureNode( 'diffuseColor' );
	const scenePassDepth = scenePass.getTextureNode( 'depth' );
	const scenePassNormal = scenePass.getTextureNode( 'normal' );
	const scenePassVelocity = scenePass.getTextureNode( 'velocity' );

	const diffuseTexture = scenePass.getTexture( 'diffuseColor' );
	diffuseTexture.type = THREE.UnsignedByteType;
	const normalTexture = scenePass.getTexture( 'normal' );
	normalTexture.type = THREE.UnsignedByteType;

	const sceneNormal = sample( ( uv ) => {

		return colorToDirection( scenePassNormal.sample( uv ) );

	} );

	const giPass = ssgi( scenePassColor, scenePassDepth, sceneNormal, camera );
	giPass.sliceCount.value = 2;
	giPass.stepCount.value = 8;

	const gi = giPass.rgb;
	const ao = giPass.a;

	const compositePass = vec4(
		add( scenePassColor.rgb.mul( ao ), scenePassDiffuse.rgb.mul( gi ) ),
		scenePassColor.a
	);

	const traaPass = traa( compositePass, scenePassDepth, scenePassVelocity, camera );
	renderPipeline.outputNode = traaPass;

	// Lights

	mouseLight = new THREE.PointLight( DEMON_PALETTE.orange, 58 );
	mouseLight.position.set( 0, BOX_HEIGHT / 2, LIGHT_DEPTH );
	mouseLight.castShadow = true;
	mouseLight.shadow.mapSize.set( 1024, 1024 );
	mouseLight.shadow.radius = 20;
	scene.add( mouseLight );

	rimLight = new THREE.PointLight( DEMON_PALETTE.teal, 18 );
	rimLight.position.set( - boxSize.w * 0.34, boxSize.h * 0.72, - BOX_DEPTH * 0.16 );
	scene.add( rimLight );

	lowLight = new THREE.PointLight( DEMON_PALETTE.coral, 10 );
	lowLight.position.set( boxSize.w * 0.26, boxSize.h * 0.16, LIGHT_DEPTH * 0.2 );
	scene.add( lowLight );

	// Raycaster (used to steer the light)

	raycaster = new THREE.Raycaster();
	pointer = new THREE.Vector2();

	// Build the box + fit the camera

	rebuildWalls();

	// Render the box right away while the face + tracking load

	renderer.setAnimationLoop( animate );

	// Load the face model, then start webcam tracking

	loadFace();
	await initFaceTracking();

	// Events

	renderer.domElement.addEventListener( 'pointermove', onPointerMove );
	window.addEventListener( 'resize', onResize );

}

// ── Cornell box ──

function rebuildWalls() {

	const disposedMaterials = new Set();
	for ( const mesh of wallMeshes ) {

		scene.remove( mesh );
		mesh.geometry.dispose();
		const materials = Array.isArray( mesh.material ) ? mesh.material : [ mesh.material ];
		for ( const material of materials ) {

			if ( ! disposedMaterials.has( material ) ) {

				material.dispose();
				disposedMaterials.add( material );

			}

		}

	}

	wallMeshes = [];

	// Box width depends on the viewport aspect ratio
	boxSize.w = getBoxWidth();

	fitCameraToBox();

	createBox();
	if ( rimLight ) rimLight.position.set( - boxSize.w * 0.34, boxSize.h * 0.72, - BOX_DEPTH * 0.16 );
	if ( lowLight ) lowLight.position.set( boxSize.w * 0.26, boxSize.h * 0.16, LIGHT_DEPTH * 0.2 );

	// Re-fit the face for the new aspect ratio (no-op until it has loaded)
	fitFace();

}

function createBox() {

	const hw = boxSize.w / 2;
	const hh = boxSize.h / 2;
	const hd = boxSize.d / 2;
	const t = WALL_THICKNESS;

	const floorMaterial = new THREE.MeshPhysicalMaterial( {
		color: DEMON_PALETTE.panel,
		roughness: 0.78,
		metalness: 0.0,
		emissive: 0x05050a,
		emissiveIntensity: 0.35,
	} );

	const backMaterial = new THREE.MeshPhysicalMaterial( {
		color: 0x151520,
		roughness: 0.82,
		metalness: 0.0,
		emissive: DEMON_PALETTE.frame,
		emissiveIntensity: 0.3,
	} );

	const leftMaterial = new THREE.MeshPhysicalMaterial( {
		color: 0x183033,
		roughness: 0.76,
		metalness: 0.0,
		emissive: DEMON_PALETTE.teal,
		emissiveIntensity: 0.08,
	} );

	const rightMaterial = new THREE.MeshPhysicalMaterial( {
		color: 0x3a1812,
		roughness: 0.76,
		metalness: 0.0,
		emissive: DEMON_PALETTE.coral,
		emissiveIntensity: 0.06,
	} );

	const walls = [
		// floor
		{ size: [ boxSize.w, t, boxSize.d ], pos: [ 0, - t / 2, 0 ], mat: floorMaterial },
		// ceiling
		{ size: [ boxSize.w, t, boxSize.d ], pos: [ 0, boxSize.h + t / 2, 0 ], mat: floorMaterial },
		// back
		{ size: [ boxSize.w, boxSize.h, t ], pos: [ 0, hh, - hd - t / 2 ], mat: backMaterial },
		// left
		{ size: [ t, boxSize.h, boxSize.d ], pos: [ - hw - t / 2, hh, 0 ], mat: leftMaterial },
		// right
		{ size: [ t, boxSize.h, boxSize.d ], pos: [ hw + t / 2, hh, 0 ], mat: rightMaterial },
	];

	for ( const w of walls ) {

		const geo = new THREE.BoxGeometry( w.size[ 0 ], w.size[ 1 ], w.size[ 2 ] );
		const mesh = new THREE.Mesh( geo, w.mat );
		mesh.position.set( ...w.pos );
		mesh.receiveShadow = true;
		scene.add( mesh );
		wallMeshes.push( mesh );

	}

}

// ── Face ──

function loadFace() {

	const ktx2Loader = new KTX2Loader()
		.setTranscoderPath( 'https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/libs/basis/' )
		.detectSupport( renderer );

	new GLTFLoader()
		.setKTX2Loader( ktx2Loader )
		.setMeshoptDecoder( MeshoptDecoder )
		.load( 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r184/examples/models/gltf/facecap.glb', ( gltf ) => {

			const mesh = gltf.scene.children[ 0 ];

			// Lit clay so the face is shaded by the box light + SSGI bounce
			faceMaterial = new THREE.MeshStandardMaterial( {
				color: DEMON_PALETTE.clay,
				roughness: 0.58,
				metalness: 0.0,
				emissive: DEMON_PALETTE.orange,
				emissiveIntensity: 0.015,
				side: THREE.DoubleSide,
			} );

			const head = mesh.getObjectByName( 'mesh_2' );
			head.material = faceMaterial;
			head.castShadow = true;
			head.receiveShadow = true;

			const teeth = mesh.getObjectByName( 'mesh_3' );
			teeth.material = faceMaterial;
			teeth.receiveShadow = true;

			face = head;
			eyeL = mesh.getObjectByName( 'eyeLeft' );
			eyeR = mesh.getObjectByName( 'eyeRight' );
			faceTransform = mesh.getObjectByName( 'grp_transform' );

			// Mirror horizontally for a selfie-like feel (localized to the face,
			// so the box's red/green walls stay put). three.js fixes the winding.
			mesh.scale.x *= - 1;

			scene.add( mesh );

			// Measure the face at its base (mirrored) scale, then fit it to the box.
			// We keep the base measurements so we can re-fit on resize / orientation change.
			scene.updateMatrixWorld( true );

			const bbox = new THREE.Box3().setFromObject( mesh );
			faceRoot = mesh;
			faceBaseScale = mesh.scale.clone();
			faceBaseSize = bbox.getSize( new THREE.Vector3() );
			faceBaseCenter = bbox.getCenter( new THREE.Vector3() );

			fitFace();

		} );

}

function fitFace() {

	if ( ! faceRoot ) return;

	// Size the face relative to the smaller viewport dimension so it doesn't
	// dominate in portrait. Landscape (aspect >= 1) keeps the full size.
	const aspect = window.innerWidth / window.innerHeight;
	const aspectFill = Math.min( 1, aspect );

	const s = ( boxSize.h * FACE_FILL * 3 * aspectFill ) / faceBaseSize.y;
	faceFitScale = s;

	faceRoot.scale.copy( faceBaseScale ).multiplyScalar( s );
	faceRoot.position.set(
		- faceBaseCenter.x * s,
		boxSize.h / 2 - faceBaseCenter.y * s,
		- faceBaseCenter.z * s
	);

}

async function initFaceTracking() {

	video = document.createElement( 'video' );
	video.muted = true;
	video.playsInline = true;

	const filesetResolver = await FilesetResolver.forVisionTasks(
		'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
	);

	faceLandmarker = await FaceLandmarker.createFromOptions( filesetResolver, {
		baseOptions: {
			modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
			delegate: 'GPU'
		},
		outputFaceBlendshapes: true,
		outputFacialTransformationMatrixes: true,
		runningMode: 'VIDEO',
		numFaces: 1
	} );

	if ( navigator.mediaDevices && navigator.mediaDevices.getUserMedia ) {

		navigator.mediaDevices.getUserMedia( { video: { facingMode: 'user' } } )
			.then( function ( stream ) {

				video.srcObject = stream;
				video.play();

			} )
			.catch( function ( error ) {

				console.error( 'Unable to access the camera/webcam.', error );

			} );

	}

}

// ── Light interaction ──

function onPointerMove( event ) {

	pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

	raycaster.setFromCamera( pointer, camera );

	// Steer the point light across the front face plane of the box
	const frontPlane = new THREE.Plane( new THREE.Vector3( 0, 0, 1 ), - LIGHT_DEPTH );
	const hit = new THREE.Vector3();
	if ( raycaster.ray.intersectPlane( frontPlane, hit ) ) {

		mouseLightTarget.copy( hit );

	}

}

// ── Camera fitting ──

function fitCameraToBox() {

	const aspect = window.innerWidth / window.innerHeight;
	const vFov = THREE.MathUtils.degToRad( CAM_FOV / 2 );

	const dist = ( boxSize.h / 2 ) / Math.tan( vFov );

	camera.aspect = aspect;
	camera.position.set( 0, boxSize.h / 2, dist + boxSize.d / 2 );
	camera.lookAt( 0, boxSize.h / 2, 0 );
	camera.updateProjectionMatrix();

}

// ── Resize ──

function onResize() {

	renderer.setSize( window.innerWidth, window.innerHeight );
	rebuildWalls();

}

// ── Animate ──

const timer = new THREE.Timer();

function animate() {

	timer.update();
	const dt = Math.min( timer.getDelta(), 1 / 30 );
	audioBloom += ( targetBloom - audioBloom ) * ( 1 - Math.exp( - 9 * dt ) );

	// Ease the point light toward the cursor
	const easeFactor = 1 - Math.exp( - EASE_SPEED * dt );
	mouseLight.position.lerp( mouseLightTarget, easeFactor );
	mouseLight.intensity = 46 + audioBloom * 100;
	rimLight.intensity = 14 + audioBloom * 42;
	lowLight.intensity = 8 + audioBloom * 28;
	renderer.toneMappingExposure = 0.42 + audioBloom * 0.16;

	if ( faceRoot ) {

		faceRoot.scale.copy( faceBaseScale ).multiplyScalar( faceFitScale * ( 1 + audioBloom * 0.026 ) );

	}

	if ( faceMaterial ) {

		faceMaterial.emissiveIntensity = 0.015 + audioBloom * 0.09;

	}

	// Drive the face from the webcam
	if ( faceLandmarker && face && video && video.readyState >= HTMLMediaElement.HAVE_METADATA ) {

		const results = faceLandmarker.detectForVideo( video, performance.now() );

		// Head pose
		if ( results.facialTransformationMatrixes.length > 0 && faceTransform ) {

			const matrix = results.facialTransformationMatrixes[ 0 ].data;

			transform.matrix.fromArray( matrix );
			transform.matrix.decompose( transform.position, transform.quaternion, transform.scale );

			faceTransform.position.x = transform.position.x;
			faceTransform.position.y = transform.position.z + 40;
			faceTransform.position.z = - transform.position.y;

			faceTransform.rotation.x = transform.rotation.x;
			faceTransform.rotation.y = transform.rotation.z;
			faceTransform.rotation.z = - transform.rotation.y;

		}

		// Blendshapes (expressions + eye gaze)
		if ( results.faceBlendshapes.length > 0 ) {

			const faceBlendshapes = results.faceBlendshapes[ 0 ].categories;

			// Eyes have no morph targets, so map their blendshape scores to rotation
			const eyeScore = {
				leftHorizontal: 0,
				rightHorizontal: 0,
				leftVertical: 0,
				rightVertical: 0,
			};

			for ( const blendshape of faceBlendshapes ) {

				const categoryName = blendshape.categoryName;
				const score = blendshape.score;

				const index = face.morphTargetDictionary[ blendshapesMap[ categoryName ] ];

				if ( index !== undefined ) {

					face.morphTargetInfluences[ index ] = score;

				}

				// Two blendshapes per axis (up/down, in/out): add one, subtract the
				// other for a final score in the -1 to 1 range
				switch ( categoryName ) {

					case 'eyeLookInLeft':
						eyeScore.leftHorizontal += score;
						break;
					case 'eyeLookOutLeft':
						eyeScore.leftHorizontal -= score;
						break;
					case 'eyeLookInRight':
						eyeScore.rightHorizontal -= score;
						break;
					case 'eyeLookOutRight':
						eyeScore.rightHorizontal += score;
						break;
					case 'eyeLookUpLeft':
						eyeScore.leftVertical -= score;
						break;
					case 'eyeLookDownLeft':
						eyeScore.leftVertical += score;
						break;
					case 'eyeLookUpRight':
						eyeScore.rightVertical -= score;
						break;
					case 'eyeLookDownRight':
						eyeScore.rightVertical += score;
						break;

				}

			}

			eyeL.rotation.z = eyeScore.leftHorizontal * eyeRotationLimit;
			eyeR.rotation.z = eyeScore.rightHorizontal * eyeRotationLimit;
			eyeL.rotation.x = eyeScore.leftVertical * eyeRotationLimit;
			eyeR.rotation.x = eyeScore.rightVertical * eyeRotationLimit;

		}

	}

	renderPipeline.render();

}
