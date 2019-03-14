function [res,dist] = observation_score(response,response_1,response_2,response_3,response_4, particles, target_size, nScales)

    loc = floor([particles(2,:)' particles(1,:)']);% [y x]
    sc  = particles(3,:)';
    sample_num = size(particles, 2);
    sr  = [particles(6,:)' ones(sample_num, 1)]; 
    sz = floor(sc.*sr.*target_size(2)); 
    score = zeros(nScales, sample_num);
    score_update = zeros(nScales, sample_num);
    for warp_num = 1:sample_num
        xs = loc(warp_num,2) + (1:sz(warp_num, 2)) - floor(sz(warp_num, 2)/2);
        ys = loc(warp_num,1) + (1:sz(warp_num, 1)) - floor(sz(warp_num, 1)/2);
        wimg1 = response(ys(ys>=1 & ys<=size(response,1)), xs(xs>=1 & xs<=size(response,2)), :);
        %find four peaks in the holistic map
        peak_1 = response_1(ys(ys>=1 & ys<=size(response_1,1)), xs(xs>=1 & xs<=size(response_1,2)), :);
        peak_2 = response_2(ys(ys>=1 & ys<=size(response_2,1)), xs(xs>=1 & xs<=size(response_2,2)), :);
        peak_3 = response_4(ys(ys>=1 & ys<=size(response_4,1)), xs(xs>=1 & xs<=size(response_4,2)), :);
        peak_4 = response_3(ys(ys>=1 & ys<=size(response_3,1)), xs(xs>=1 & xs<=size(response_3,2)), :);
        
%         [~,I_1] = max(peak_1(:));
%         [y1, x1] = ind2sub(size(peak_1),I_1);
%         [~,I_2] = max(peak_2(:));
%         [y2, x2] = ind2sub(size(peak_2),I_2);
%         [~,I_3] = max(peak_3(:));
%         [y3, x3] = ind2sub(size(peak_3),I_3);
%         [~,I_4] = max(peak_4(:));
%         [y4, x4] = ind2sub(size(peak_4),I_4);

        [y1, x1] = find(peak_1==max(max(peak_1)));
        [y2, x2] = find(peak_2==max(max(peak_2)));
        [y3, x3] = find(peak_3==max(max(peak_3)));
        [y4, x4] = find(peak_4==max(max(peak_4)));
        if isempty(y1)
            y1=0;
        end
        if isempty(x1)
            x1=0;
        end
        if isempty(y2)
            y2=0;
        end
        if isempty(x2)
            x2=0;
        end       
        if isempty(x3)
            x3=0;
        end
        if isempty(y3)
            y3=0;
        end
        if isempty(x4)
            x4=0;
        end
        if isempty(y4)
            y4=0;
        end          
        y1=y1(1,1);
        x1=x1(1,1);
        y2=y2(1,1);
        x2=x2(1,1);
        y3=y3(1,1);
        x3=x3(1,1);
        y4=y4(1,1);
        x4=x4(1,1);
        %initialize the particle's holistic map
        u1=0;
        v1=0;
        u3=target_size(2);
        v3=target_size(1);
        u4=0;
        v4=v3;
        u2=u3;
        v2=0;
        ss=u3*v3;      
        %four center points counterclockwise form four triangle
        e1=sqrt((x1-u1)^2+(y1-v1)^2);
        e2=sqrt((x1-u4)^2+(y1-v4)^2);
        e3=sqrt((u1-u4)^2+(v1-v4)^2);
        pe=(e1+e2+e3)/2;
        fe = istriangle(e1, e2, e3);
        if fe == true
            se=sqrt(pe*(pe-e1)*(pe-e2)*(pe-e3))/ss;
        else
            se=0;
        end
        
        f1=sqrt((x2-u1)^2+(y2-v1)^2);
        f2=sqrt((x2-u2)^2+(y2-v2)^2);
        f3=sqrt((u1-u2)^2+(v1-v2)^2);
        pf=(f1+f2+f3)/2;
        ff = istriangle(f1, f2, f3);
        if ff == true
            sf=sqrt(pf*(pf-f1)*(pf-f2)*(pf-f3))/ss;
        else
            sf=0;
        end
        
        g1=sqrt((x3-u2)^2+(y3-v2)^2);
        g2=sqrt((x3-u3)^2+(y3-v3)^2);
        g3=sqrt((u2-u3)^2+(v2-v3)^2);
        pg=(g1+g2+g3)/2;
        fg = istriangle(g1, g2, g3);
        if fg == true
            sg=sqrt(pg*(pg-g1)*(pg-g2)*(pg-g3))/ss;
        else
            sg=0;
        end       
        
        h1=sqrt((x4-u3)^2+(y4-v3)^2);
        h2=sqrt((x4-u4)^2+(y4-v4)^2);
        h3=sqrt((u3-u4)^2+(v3-v4)^2);
        ph=(h1+h2+h3)/2;
        fh = istriangle(h1, h2, h3);
        if fh == true
            sh=sqrt(ph*(ph-h1)*(ph-h2)*(ph-h3))/ss;
        else
            sh=0;
        end
        %distance measuring the scale similarity
        d_sca=(sqrt((se-0.125)^2+(sf-0.125)^2+(sg-0.125)^2+(sh-0.125)^2));
        
        %calculate the area of the rectangle
        m1=sqrt((x3-x1)^2+(y3-y1)^2);
        m2=sqrt((x3-x4)^2+(y3-y4)^2);
        m3=sqrt((x4-x1)^2+(y4-y1)^2);
        p5=(m1+m2+m3)/2;
        s5=sqrt(p5*(p5-m1)*(p5-m2)*(p5-m3));
        
        n1=sqrt((x3-x1)^2+(y3-y1)^2);
        n2=sqrt((x3-x2)^2+(y3-y2)^2);
        n3=sqrt((x2-x1)^2+(y2-y1)^2);
        p6=(n1+n2+n3)/2;
        s6=sqrt(p6*(p6-n1)*(p6-n2)*(p6-n3));
        s=s5+s6;
        
        %four center points counterclockwise form four triangle
        centre=target_size./2;
        y0=centre(:,1);
        x0=centre(:,2);
        a1=sqrt((x0-x1)^2+(y0-y1)^2);
        a2=sqrt((x0-x2)^2+(y0-y2)^2);
        a3=sqrt((x2-x1)^2+(y2-y1)^2);
        p1=(a1+a2+a3)/2;
        fa = istriangle(a1, a2, a3);
        if fa == true
            s1=sqrt(p1*(p1-a1)*(p1-a2)*(p1-a3))/s;
        else
            s1=0;
        end 
        
        b1=sqrt((x0-x2)^2+(y0-y2)^2);
        b2=sqrt((x0-x3)^2+(y0-y3)^2);
        b3=sqrt((x2-x3)^2+(y2-y3)^2);
        p2=(b1+b2+b3)/2;
        fb = istriangle(b1, b2, b3);
        if fb == true
            s2=sqrt(p2*(p2-b1)*(p2-b2)*(p2-b3))/s;
        else
            s2=0;
        end 
        
        c1=sqrt((x0-x3)^2+(y0-y3)^2);
        c2=sqrt((x0-x4)^2+(y0-y4)^2);
        c3=sqrt((x3-x4)^2+(y3-y4)^2);
        p3=(c1+c2+c3)/2;
        fc = istriangle(c1, c2, c3);
        if fc == true
            s3=sqrt(p3*(p3-c1)*(p3-c2)*(p3-c3))/s;
        else
            s3=0;
        end 
        
        d1=sqrt((x0-x1)^2+(y0-y1)^2);
        d2=sqrt((x0-x4)^2+(y0-y4)^2);
        d3=sqrt((x4-x1)^2+(y4-y1)^2);
        p4=(d1+d2+d3)/2;
        fd = istriangle(d1, d2, d3);
        if fd == true
            s4=sqrt(p4*(p4-d1)*(p4-d2)*(p4-d3))/s;
        else
            s4=0;
        end 
        
        d_loc=(sqrt((s1-0.25)^2+(s2-0.25)^2+(s3-0.25)^2+(s4-0.25)^2));
        
        %d=exp(-1*300*(d_loc));

        d=exp(-1*300*(d_loc));
        score(:, warp_num)=d;
        %score(:, warp_num) = sum(sum(wimg1, 1), 2)./prod(sz(warp_num, :)); %
    end
     dist=score;
     score_w=score./sum(score(1,:));
%      [temp,Ix]=sort(score_w,2);
%      new_score=temp(:,290:end);
%      new_loc=loc(Ix(:,290:end),:);
%      new_sc=sc(Ix(:,290:end),:);
    [~,i]=max(score_w);
    res.pos = loc(i,:);
    res.scale = sc(i,:);
%     [~, sc_ind] = max(sum(new_score, 2));
%     res.pos = sum(new_loc.*new_score(sc_ind,:)')/sum(new_score(sc_ind,:));
%     res.scale = sum(new_sc.*new_score(sc_ind,:)')/sum(new_score(sc_ind,:));

end