import {test,expect} from '@playwright/test';

test('desktop UI invokes the real fitted API and displays its output',async({page,request})=>{
 await page.goto('/');
 await expect(page.getByText('MODEL ONLINE',{exact:true})).toBeVisible();
 await expect(page.locator('input,select')).toHaveCount(6);
 const actual=request.post('http://localhost:8000/predict',{data:{Player_Age:28,Player_Weight:75,Player_Height:180,Previous_Injuries:0,Training_Intensity:.5,Recovery_Time:4}});
 const responsePromise=page.waitForResponse(r=>r.url().endsWith('/predict')&&r.request().method()==='POST');
 await page.getByRole('button',{name:/RUN PREDICTION/}).click();
 const response=await responsePromise;
 expect(response.status()).toBe(200);
 const result=await response.json();
 expect(result).toEqual(await (await actual).json());
 await expect(page.getByRole('heading',{name:result.label,exact:true})).toBeVisible();
 await expect(page.locator('.gauge strong')).toContainText((result.injury_probability*100).toFixed(1));
 await page.screenshot({path:'../reports/frontend-desktop.png',fullPage:true});
});

test('mobile form remains usable and validates out-of-range values',async({page})=>{
 await page.setViewportSize({width:390,height:844});
 await page.goto('/');
 await expect(page.getByText('MODEL ONLINE',{exact:true})).toBeVisible();
 await page.getByRole('button',{name:/RUN PREDICTION/}).click();
 await expect(page.locator('.gauge')).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 await page.screenshot({path:'../reports/frontend-mobile.png',fullPage:true});
 await page.locator('input').first().fill('99');
 expect(await page.locator('input').first().evaluate(el=>el.validity.rangeOverflow)).toBe(true);
});
