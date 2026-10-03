import fs from 'node:fs';
import path from 'node:path';
import {initNodeBackend} from '../packages/tiny-llm/src/cli/node-backend.js';
import {loadModelDir,saveModelDir} from '../packages/tiny-llm/src/cli/files.js';
import {Trainer,encodeLines,generate,disposeModel} from '../packages/tiny-llm/src/index.js';
const candidates=fs.readdirSync('models').filter(n=>fs.existsSync(`models/${n}/meta.json`)).map(n=>({n,m:JSON.parse(fs.readFileSync(`models/${n}/meta.json`,'utf8'))})).sort((a,b)=>Date.parse(b.m.exportedAt)-Date.parse(a.m.exportedAt));
const source=process.argv[2] ?? (candidates[0] ? `models/${candidates[0].n}` : '');
if (!source) throw new Error('No checkpoint found. Supply a model directory.');
const budget=Number(process.argv[3] ?? 1000);
if (!Number.isSafeInteger(budget) || budget<1) throw new Error('Steps must be a positive integer');
const out=`models/chat-evaluation-${Date.now()}`;
console.log({source,out,backend:await initNodeBackend()});
const model=loadModelDir(source);
const lines=fs.readFileSync('train-data/vi-chat-basic.txt','utf8').trim().split('\n');
const questions=['xin chào','bạn là ai','bạn khỏe không','chào bạn','bạn tên gì','cảm ơn','chào nhé','bạn có khỏe không'];
const report: {source:string; config:typeof model.config; trainingLearningRate:number; corpus:string[]; checkpoints:{step:number; loss?:number; answers:{question:string; answer:string}[]}[]}={source,config:model.config,trainingLearningRate:0.0003,corpus:lines,checkpoints:[]};
const sourceMeta=fs.existsSync(path.join(source,'meta.json')) ? JSON.parse(fs.readFileSync(path.join(source,'meta.json'),'utf8')) : {step:0};
function saveCheckpoint(step:number,loss:number){
  saveModelDir(model,out);
  fs.writeFileSync(path.join(out,'meta.json'),JSON.stringify({...sourceMeta,step:(sourceMeta.step ?? 0)+step,loss,valLoss:-1,valAcc:-1,exportedAt:new Date().toISOString(),label:'chat-evaluation',viCurriculum:sourceMeta.viCurriculum ? {...sourceMeta.viCurriculum,phase:'chat',chatSteps:(sourceMeta.viCurriculum.chatSteps ?? 0)+step,plannedEndStep:(sourceMeta.step ?? 0)+budget}:undefined},null,2));
}
function evaluate(step:number,loss?:number){const answers=questions.map(question=>({question,answer:generate(model,`Người dùng: ${question} = Trợ lý:`,{temperature:0,maxNewTokens:90}).text})); report.checkpoints.push({step,loss,answers}); fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'evaluation.json'),JSON.stringify(report,null,2)); console.log(JSON.stringify({step,loss,answers}));}
evaluate(0);
model.config.learningRate=0.0003;
const trainer=new Trainer(model,encodeLines(model.tokenizer,lines),new Int32Array());
for(let step=1;step<=budget;step++){const loss=trainer.trainStep();if(step%25===0)console.log(`step ${step} loss ${loss}`);if(step%250===0 || step===budget){evaluate(step,loss);saveCheckpoint(step,loss);}}
trainer.dispose();disposeModel(model);
