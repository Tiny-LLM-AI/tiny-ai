import fs from 'node:fs';
import {initNodeBackend} from '../packages/tiny-llm/src/cli/node-backend.js';
import {loadModelDir} from '../packages/tiny-llm/src/cli/files.js';
import {generate,disposeModel} from '../packages/tiny-llm/src/index.js';
const dir=process.argv[2];
await initNodeBackend();
const model=loadModelDir(dir);
const probes=['xin chào','bạn là ai','bạn khỏe không','xin chào bạn','Bạn là ai?','cho mình biết bạn là ai','hôm nay bạn khỏe không','chào nhé','tên của bạn là gì','bạn bao nhiêu tuổi','2 cộng 3 bằng bao nhiêu?'];
const results=probes.map(question=>({question,answer:generate(model,`Người dùng: ${question} = Trợ lý:`,{temperature:0,maxNewTokens:64}).text}));
const sampled=['xin chào','bạn là ai','bạn khỏe không'].flatMap(question=>[1,2,3].map(seed=>({question,seed,answer:generate(model,`Người dùng: ${question} = Trợ lý:`,{temperature:0.2,topK:20,seed,maxNewTokens:64}).text})));
let history='';const conversation=[];
for(const question of ['xin chào','bạn là ai','bạn khỏe không']) {history+=`Người dùng: ${question} = Trợ lý:`;const answer=generate(model,history,{temperature:0.2,topK:20,seed:1,maxNewTokens:64}).text;history+=answer+' ';conversation.push({question,answer});}
fs.writeFileSync(`${dir}/probes.json`,JSON.stringify({results,sampled,conversation},null,2));console.log(JSON.stringify({results,sampled,conversation},null,2));disposeModel(model);
